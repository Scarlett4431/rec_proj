import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data_loader import load_movielens_1m
from src.features.user_features import build_user_features
from src.features.item_features import build_item_features
from src.features.feature_store import FeatureStore
from src.recall_dataset import RecallDataset
from src.rank_dataset import RankDataset
from src.recall.two_tower import TwoTowerModel
from src.losses import InBatchSoftmaxLoss
from src.rank.rank_mlp import RankerMLP
from src.evaluation import evaluate_with_faiss, evaluate_ranker_with_faiss
from src.utils import remap_ids, user_stratified_split
from src.faiss_index import FaissIndex


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
    user_feats_df = build_user_features(train)
    item_feats_df = build_item_features(train, movies)

    user_store = FeatureStore(
        user_feats_df,
        "user_idx",
        numeric_cols=[],
        cat_cols=[],
        bucket_cols=["user_total_ratings", "user_avg_rating", "user_recency_days"],
        bucket_bins=10,
    )
    item_numeric_cols = ["item_total_ratings", "item_avg_rating", "item_release_year"]
    item_store = FeatureStore(
        item_feats_df,
        "item_idx",
        numeric_cols=[],
        cat_cols=[],
        bucket_cols=item_numeric_cols,
        bucket_bins=10,
    )

    # Precompute dense feature matrices (CPU + device copies)
    user_feat_matrix_cpu = user_store.to_matrix(max_user_id)
    item_feat_matrix_cpu = item_store.to_matrix(max_item_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_feat_matrix = user_feat_matrix_cpu.to(device)
    item_feat_matrix = item_feat_matrix_cpu.to(device)

    # -----------------------------
    # 4) Recall dataset & loader (warm-up)
    # -----------------------------
    recall_ds = RecallDataset(train, user_store, item_store)
    recall_loader = DataLoader(recall_ds, batch_size=1024, shuffle=True, drop_last=True)

    model = TwoTowerModel(
        num_users=num_users,
        num_items=num_items,
        embed_dim=64,
        user_extra_dim=recall_ds.user_feat_dim,
        item_extra_dim=recall_ds.item_feat_dim,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
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
            u_feats = batch["user_feats"].to(device)
            i_feats = batch["pos_item_feats"].to(device)

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
        with torch.no_grad():
            user_ids_full = torch.arange(num_users, device=device, dtype=torch.long)
            item_ids_full = torch.arange(num_items, device=device, dtype=torch.long)
            u_emb = model.user_embed(user_ids_full, user_feats=user_feat_matrix)
            i_emb = model.item_embed(item_ids_full, item_feats=item_feat_matrix)
        return u_emb, i_emb

    user_emb, item_emb = export_embeddings()

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
            u_feats = batch["user_feats"].to(device)
            pos_feats = batch["pos_item_feats"].to(device)

            u_emb, pos_emb = model(user_ids, pos_items,
                                   user_feats=u_feats,
                                   item_feats=pos_feats)
            v_emb = pos_emb

            hard_items = batch["hard_neg_items"].to(device)
            if hard_items.numel() > 0:
                valid_mask = hard_items >= 0
                if valid_mask.any():
                    neg_item_ids = hard_items[valid_mask]
                    neg_item_feats = item_feat_matrix[neg_item_ids]
                    hard_emb = model.item_embed(neg_item_ids, item_feats=neg_item_feats)
                    v_emb = torch.cat([v_emb, hard_emb], dim=0)

            opt.zero_grad()
            loss = recall_loss(u_emb, v_emb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[Recall+HardNeg] Epoch {epoch + 1}, Loss = {total_loss / len(recall_hard_loader):.4f}")

    user_emb, item_emb = export_embeddings()

    # Build recall candidate pool for ranking using FAISS top-k
    candidate_k = 200
    faiss_index = FaissIndex(item_emb.detach().cpu())
    user_candidates = {}
    for u in range(num_users):
        _, idxs = faiss_index.search(user_emb[u].detach().cpu(), k=candidate_k)
        candidates = [int(i) for i in idxs.tolist() if i >= 0]
        user_candidates[u] = candidates

    # -----------------------------
    # 7) Recall evaluation
    # -----------------------------
    recall_results = evaluate_with_faiss(user_emb.cpu(), item_emb.cpu(), test_pairs, k=200)
    print("FAISS Recall Eval (after hard-neg):", recall_results)

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
    # for idx in range(min(200, len(rank_train_ds))):
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
        user_feat_dim=user_store.total_dim,
        item_feat_dim=item_store.total_dim,
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
            faiss_k=200,
            rank_k=10,
            device=device,
            user_feat_matrix=user_feat_matrix_cpu,
            item_feat_matrix=item_feat_matrix_cpu,
        )
    print("Ranker Eval (Recall+Rank):", rank_results)


if __name__ == "__main__":
    main()
