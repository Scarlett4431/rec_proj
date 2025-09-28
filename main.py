import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_loader import load_movielens_1m
from src.features.user_features import build_user_features
from src.features.item_features import build_item_features
from src.features.feature_store import FeatureStore
from src.feature_encoder import FeatureEncoder
from src.recall_dataset import RecallDataset
from src.rank_dataset import RankDataset
from src.recall.two_tower import TwoTowerModel
from src.recall.covisitation import build_covisitation_index, build_user_covisitation_candidates
from src.recall.item_cf import build_item_cf_index, build_user_itemcf_candidates
from src.recall.hybrid import merge_candidate_lists
from src.losses import InBatchSoftmaxLoss
from src.rank.rank_mlp import RankerMLP
from src.evaluation import evaluate_with_faiss, evaluate_ranker_with_faiss
from src.utils import remap_ids, user_stratified_split
from src.faiss_index import FaissIndex


def move_to_device(struct, device):
    if isinstance(struct, torch.Tensor):
        return struct.to(device)
    if isinstance(struct, dict):
        return {k: move_to_device(v, device) for k, v in struct.items()}
    return struct


def encode_features(encoder, feature_batch, device):
    feature_batch = move_to_device(feature_batch, device)
    return encoder(feature_batch)


def encode_store_batch(store, encoder, ids, device, max_lengths):
    if isinstance(ids, torch.Tensor):
        id_list = ids.detach().cpu().tolist()
    else:
        id_list = ids
    batch = store.get_batch(id_list, max_multi_lengths=max_lengths)
    return encode_features(encoder, batch, device)


def encode_all_entities(store, encoder, count, device, max_lengths):
    ids = list(range(count))
    return encode_store_batch(store, encoder, ids, device, max_lengths)


def main():
    # -----------------------------
    # 1) Load data & remap IDs
    # -----------------------------
    ratings, movies = load_movielens_1m("data/ml-1m")
    ratings, movies, user2id, item2id = remap_ids(ratings, movies)

    max_user_id = int(ratings["user_idx"].max())
    max_item_id = int(ratings["item_idx"].max())
    num_users = max_user_id + 1
    num_items = max_item_id + 1

    # -----------------------------
    # 2) Train/test split (avoid cold-start users in test)
    # -----------------------------
    train, test = user_stratified_split(ratings, test_frac=0.1, random_state=42)
    test_pairs = list(zip(test.user_idx.values.tolist(), test.item_idx.values.tolist()))

    # -----------------------------
    # 3) Build engineered features from train slice
    # -----------------------------
    user_feats_df = build_user_features(train, movies)
    item_feats_df = build_item_features(train, movies)

    # covis_index = build_covisitation_index(train, max_items_per_user=50, top_k=200)
    item_cf_index = build_item_cf_index(train, max_items_per_user=100, top_k=200)

    user_store = FeatureStore(
        user_feats_df,
        "user_idx",
        numeric_cols=[],
        cat_cols=["temporal_preference"],
        multi_cat_cols=["watched_items", "favorite_genres"],
        bucket_cols=["user_total_ratings", "user_avg_rating", "user_recency_days", "user_avg_release_year"],
        bucket_bins=10,
    )
    item_numeric_cols = ["item_total_ratings", "item_avg_rating", "item_release_year"]
    item_store = FeatureStore(
        item_feats_df,
        "item_idx",
        numeric_cols=[],
        cat_cols=[],
        multi_cat_cols=["item_genres"],
        bucket_cols=item_numeric_cols,
        bucket_bins=10,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_encoder = FeatureEncoder(
        numeric_dim=user_store.numeric_dim,
        cat_dims=user_store.cat_dims,
        bucket_dims=user_store.bucket_dims,
        multi_cat_dims=user_store.multi_cat_dims,
        embed_dim=32,
        proj_dim=32,
    ).to(device)

    item_encoder = FeatureEncoder(
        numeric_dim=item_store.numeric_dim,
        cat_dims=item_store.cat_dims,
        bucket_dims=item_store.bucket_dims,
        multi_cat_dims=item_store.multi_cat_dims,
        embed_dim=32,
        proj_dim=32,
    ).to(device)

    user_multi_max = {"watched_items": 50, "favorite_genres": 3}
    item_multi_max = {"item_genres": 5}

    # -----------------------------
    # 4) Recall dataset & loader (warm-up)
    # -----------------------------
    recall_ds = RecallDataset(train, user_store, item_store)
    recall_loader = DataLoader(recall_ds, batch_size=1024, shuffle=True, drop_last=True)

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embed_dim=64,
        user_extra_dim=user_encoder.out_dim,
        item_extra_dim=item_encoder.out_dim,
    ).to(device)
    opt = optim.Adam(
        list(model.parameters()) + list(user_encoder.parameters()) + list(item_encoder.parameters()),
        lr=1e-3,
    )
    recall_loss = InBatchSoftmaxLoss()

    # -----------------------------
    # 5) Warm-up recall training
    # -----------------------------
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for batch in recall_loader:
            user_ids = batch["user_id"].to(device)
            pos_items = batch["pos_item"].to(device)

            u_feats = encode_store_batch(user_store, user_encoder, user_ids, device, user_multi_max)
            i_feats = encode_store_batch(item_store, item_encoder, pos_items, device, item_multi_max)

            opt.zero_grad()
            u_emb, v_emb = model(user_ids, pos_items,
                                 user_feats=u_feats,
                                 item_feats=i_feats)
            loss = recall_loss(u_emb, v_emb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[Recall] Warm-up Epoch {epoch + 1}, Loss = {total_loss / len(recall_loader):.4f}")

    def export_embeddings():
        model.eval()
        user_encoder.eval()
        item_encoder.eval()
        with torch.no_grad():
            user_ids_full = torch.arange(num_users, dtype=torch.long)
            item_ids_full = torch.arange(num_items, dtype=torch.long)

            user_feats_raw = user_store.get_batch(user_ids_full.tolist(), max_multi_lengths=user_multi_max)
            item_feats_raw = item_store.get_batch(item_ids_full.tolist(), max_multi_lengths=item_multi_max)

            user_side = encode_features(user_encoder, user_feats_raw, device)
            item_side = encode_features(item_encoder, item_feats_raw, device)

            u_emb = model.user_embed(user_ids_full.to(device), user_feats=user_side)
            i_emb = model.item_embed(item_ids_full.to(device), item_feats=item_side)
        model.train()
        user_encoder.train()
        item_encoder.train()
        return u_emb, i_emb

    user_emb, item_emb = export_embeddings()

    # covis_user_candidates = build_user_covisitation_candidates(
    #     train,
    #     covis_index,
    #     top_k=100,
    #     max_history=25,
    # )
    itemcf_user_candidates = build_user_itemcf_candidates(
        train,
        item_cf_index,
        top_k=100,
        max_history=50,
    )

    # -----------------------------
    # 6) Hard-negative mining dataset & fine-tuning
    # -----------------------------
    recall_hard_ds = RecallDataset(
        train,
        user_store,
        item_store,
        item_emb=item_emb.detach().cpu(),
        hard_neg_k=50,
        hard_neg_samples=2,
    )
    recall_hard_loader = DataLoader(recall_hard_ds, batch_size=512, shuffle=True, drop_last=True)

    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for batch in recall_hard_loader:
            user_ids = batch["user_id"].to(device)
            pos_items = batch["pos_item"].to(device)

            u_feats = encode_store_batch(user_store, user_encoder, user_ids, device, user_multi_max)
            pos_feats = encode_store_batch(item_store, item_encoder, pos_items, device, item_multi_max)

            u_emb, pos_emb = model(user_ids, pos_items,
                                   user_feats=u_feats,
                                   item_feats=pos_feats)
            v_emb = pos_emb

            hard_items = batch["hard_neg_items"].to(device)
            if hard_items.numel() > 0:
                valid_mask = hard_items >= 0
                if valid_mask.any():
                    neg_item_ids = hard_items[valid_mask]
                    neg_feats = encode_store_batch(item_store, item_encoder, neg_item_ids, device, item_multi_max)
                    hard_emb = model.item_embed(neg_item_ids, item_feats=neg_feats)
                    v_emb = torch.cat([v_emb, hard_emb], dim=0)

            opt.zero_grad()
            loss = recall_loss(u_emb, v_emb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[Recall+HardNeg] Epoch {epoch + 1}, Loss = {total_loss / len(recall_hard_loader):.4f}")

    user_emb, item_emb = export_embeddings()

    user_feat_matrix_cpu = encode_all_entities(user_store, user_encoder, num_users, device, user_multi_max).detach().cpu()
    item_feat_matrix_cpu = encode_all_entities(item_store, item_encoder, num_items, device, item_multi_max).detach().cpu()
    user_feat_matrix = user_feat_matrix_cpu.to(device)
    item_feat_matrix = item_feat_matrix_cpu.to(device)

    # Build recall candidate pool for ranking using FAISS top-k
    candidate_k = 100
    faiss_index = FaissIndex(item_emb.detach().cpu())
    user_candidates = {}
    for u in range(num_users):
        _, idxs = faiss_index.search(user_emb[u].detach().cpu(), k=candidate_k)
        faiss_candidates = [int(i) for i in idxs.tolist() if i >= 0]
        # covis_list = covis_user_candidates.get(u, [])
        itemcf_list = itemcf_user_candidates.get(u, [])
        user_candidates[u] = merge_candidate_lists(
            [
                (faiss_candidates, 0.6),
                # (covis_list, 0.3),
                (itemcf_list, 0.4),
            ],
            candidate_k,
        )

    # -----------------------------
    # 7) Recall evaluation
    # -----------------------------
    recall_results = evaluate_with_faiss(user_emb.cpu(), item_emb.cpu(), test_pairs, k=100)
    print("FAISS Recall Eval (after hard-neg):", recall_results)

    user_encoder.eval()
    item_encoder.eval()
    for param in user_encoder.parameters():
        param.requires_grad_(False)
    for param in item_encoder.parameters():
        param.requires_grad_(False)

    # -----------------------------
    # 8) Ranking dataset & model
    # -----------------------------
    rank_train_ds = RankDataset(
        ratings_df=train,
        num_items=num_items,
        user_store=user_store,
        item_store=item_store,
        num_negatives=5,
        user_emb=user_emb.detach().cpu(),
        item_emb=item_emb.detach().cpu(),
        user_feat_tensor=user_feat_matrix_cpu,
        item_feat_tensor=item_feat_matrix_cpu,
        user_candidates=user_candidates,
    )

    # # Debug: verify negatives come from recall shortlist
    # debug_mismatches = 0
    # for idx in range(min(100, len(rank_train_ds))):
    #     u, i, label = rank_train_ds.samples[idx].tolist()
    #     if label == 0 and i not in user_candidates.get(u, []):
    #         debug_mismatches += 1
    #         if debug_mismatches <= 5:
    #             print(f"[Debug] negative {i} for user {u} not in recall shortlist")
    # if debug_mismatches == 0:
    #     print("[Debug] all checked negatives come from recall candidates.")

    rank_loader = DataLoader(rank_train_ds, batch_size=512, shuffle=True, drop_last=True)

    ranker = RankerMLP(
        user_dim=user_emb.shape[1],
        item_dim=item_emb.shape[1],
        user_feat_dim=user_encoder.out_dim,
        item_feat_dim=item_encoder.out_dim,
        hidden_dim=128,
    ).to(device)

    opt_r = optim.Adam(ranker.parameters(), lr=1e-3)
    rank_loss = nn.BCELoss()

    for epoch in range(3):
        ranker.train()
        total_loss = 0.0
        for batch in rank_loader:
            u_emb_batch = batch["u_emb"].to(device)
            i_emb_batch = batch["i_emb"].to(device)
            u_feats_batch = batch["u_feats"].to(device)
            i_feats_batch = batch["i_feats"].to(device)
            labels = batch["label"].to(device)

            opt_r.zero_grad()
            preds = ranker(u_emb_batch, i_emb_batch,
                           u_feats=u_feats_batch,
                           i_feats=i_feats_batch)
            loss = rank_loss(preds, labels)
            loss.backward()
            opt_r.step()
            total_loss += loss.item()
        print(f"[Rank] Epoch {epoch + 1}, Loss = {total_loss / len(rank_loader):.4f}")

    # -----------------------------
    # 9) Ranker evaluation (recall + rerank)
    # -----------------------------
    ranker.eval()
    with torch.no_grad():
        rank_results = evaluate_ranker_with_faiss(
            ranker,
            user_emb.cpu(),
            item_emb.cpu(),
            test_pairs,
            faiss_k=100,
            rank_k=10,
            device=device,
            user_feat_matrix=user_feat_matrix_cpu,
            item_feat_matrix=item_feat_matrix_cpu,
        )
    print("Ranker Eval (Recall+Rank):", rank_results)


if __name__ == "__main__":
    main()
