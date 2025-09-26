import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# --- project imports ---
from src.data_loader import load_movielens_1m
from src.features.user_features import build_user_features
from src.features.item_features import build_item_features
from src.recall_dataset import RecallDataset
from src.rank_dataset import RankDataset
from src.recall.two_tower import TwoTowerModel
from src.losses import InBatchSoftmaxLoss
from src.rank.rank_mlp import RankerMLP
from src.evaluation import evaluate_with_faiss, evaluate_ranker_with_faiss
from src.utils import remap_ids


def main():
    # -----------------------------
    # 1) Load data
    # -----------------------------

    ratings, movies = load_movielens_1m("data/ml-1m")
    ratings, movies, user2id, item2id = remap_ids(ratings, movies)

    train, test = train_test_split(ratings, test_size=0.1, random_state=42)

    num_users = ratings["user_idx"].nunique()
    num_items = ratings["item_idx"].nunique()

    # -----------------------------
    # 2) Split train/test first (avoid leakage)
    # -----------------------------
    train, test = train_test_split(ratings, test_size=0.1, random_state=42)
    test_pairs = list(zip(test.userId.values.tolist(), test.movieId.values.tolist()))

    # -----------------------------
    # 3) Build features from train slice only
    # -----------------------------
    user_feats_df = build_user_features(train)
    item_feats_df = build_item_features(train, movies)

    # -----------------------------
    # 4) Recall dataset (first-pass, no hard negatives)
    # -----------------------------
    recall_ds = RecallDataset(train, user_feats_df, item_feats_df)
    recall_loader = DataLoader(recall_ds, batch_size=1024, shuffle=True, drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    # 5) Train recall model (warm-up)
    # -----------------------------
    # Warm-up recall training
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for batch in recall_loader:
            user_ids = batch["user_id"].to(device)
            item_ids = batch["pos_item"].to(device)
            u_feats  = batch["user_feats"].to(device)
            i_feats  = batch["pos_item_feats"].to(device)

            opt.zero_grad()
            u_emb, v_emb = model(user_ids, item_ids,
                                user_feats=u_feats,
                                item_feats=i_feats)
            loss = recall_loss(u_emb, v_emb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[Recall] Warm-up Epoch {epoch+1}, Loss = {total_loss/len(recall_loader):.4f}")

    # -----------------------------
    # 6) Export embeddings (for hard-neg mining)
    # -----------------------------
    all_user_feats = recall_ds.get_user_feature_matrix(num_users).to(device)
    all_item_feats = recall_ds.get_item_feature_matrix(num_items).to(device)

    def export_embeddings():
        model.eval()
        with torch.no_grad():
            user_ids_full = torch.arange(num_users + 1, device=device, dtype=torch.long)
            item_ids_full = torch.arange(num_items + 1, device=device, dtype=torch.long)
            u_emb = model.user_embed(user_ids_full, user_feats=all_user_feats)
            i_emb = model.item_embed(item_ids_full, item_feats=all_item_feats)
        return u_emb, i_emb

    user_emb, item_emb = export_embeddings()

    # -----------------------------
    # 7) Build recall dataset with hard negatives
    # -----------------------------
    recall_hard_ds = RecallDataset(
        train,
        user_feats_df,
        item_feats_df,
        item_emb=item_emb.detach().cpu(),
        hard_neg_k=50,
        hard_neg_samples=2
    )
    recall_hard_loader = DataLoader(recall_hard_ds, batch_size=512, shuffle=True, drop_last=True)

    # -----------------------------
    # 8) Fine-tune with hard negatives
    # -----------------------------
    for epoch in range(3):
        model.train()
        total_loss = 0.0
        for batch in recall_hard_loader:
            user_ids = batch["user_id"].to(device)           # [B]
            pos_items = batch["pos_item"].to(device)         # [B]
            u_feats = batch["user_feats"].to(device)         # [B, Fu]
            pos_feats = batch["pos_item_feats"].to(device)   # [B, Fi]

            # Positive pairs
            u_emb, pos_emb = model(user_ids, pos_items,
                                   user_feats=u_feats,
                                   item_feats=pos_feats)
            v_emb = pos_emb

            # Hard negatives (expect shape [B, S], padded with -1)
            hard_items = batch["hard_neg_items"].to(device)
            if hard_items.numel() > 0 and hard_items.dim() == 2:
                B, S = hard_items.shape
                hard_items_flat = hard_items.reshape(-1)       # [B*S]
                valid_mask = hard_items_flat >= 0
                hard_items_valid = hard_items_flat[valid_mask]

                if hard_items_valid.numel() > 0:
                    user_ids_expanded = user_ids.repeat_interleave(S)[valid_mask]
                    u_feats_expanded = u_feats.repeat_interleave(S, dim=0)[valid_mask]
                    hard_feats = all_item_feats[hard_items_valid]

                    _, hard_emb = model(user_ids_expanded,
                                        hard_items_valid,
                                        user_feats=u_feats_expanded,
                                        item_feats=hard_feats)
                    v_emb = torch.cat([v_emb, hard_emb], dim=0)

            opt.zero_grad()
            loss = recall_loss(u_emb, v_emb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"[Recall+HardNeg] Epoch {epoch+1}, Loss = {total_loss/len(recall_hard_loader):.4f}")

    # -----------------------------
    # 9) Re-export embeddings (after fine-tune)
    # -----------------------------
    user_emb, item_emb = export_embeddings()

    # Evaluate recall directly
    recall_results = evaluate_with_faiss(user_emb.cpu(), item_emb.cpu(), test_pairs, k=10)
    print("FAISS Recall Eval (after hard-neg):", recall_results)

    # -----------------------------
    # 10) Rank dataset & model
    # -----------------------------
    rank_train_ds = RankDataset(
        ratings_df=train,
        num_items=num_items,
        user_features=user_feats_df,
        item_features=item_feats_df,
        num_negatives=5,
        user_emb=user_emb.detach().cpu(),
        item_emb=item_emb.detach().cpu(),
    )
    rank_loader = DataLoader(rank_train_ds, batch_size=512, shuffle=True, drop_last=True)

    ranker = RankerMLP(
        user_dim=user_emb.shape[1],
        item_dim=item_emb.shape[1],
        extra_dim=rank_train_ds.extra_dim,
        hidden_dim=128
    ).to(device)

    opt_r = optim.Adam(ranker.parameters(), lr=1e-3)
    rank_loss = nn.BCELoss()

    for epoch in range(3):
        ranker.train()
        total_loss = 0.0
        for u_emb_batch, i_emb_batch, extra_feats, labels in rank_loader:
            u_emb_batch, i_emb_batch = u_emb_batch.to(device), i_emb_batch.to(device)
            extra_feats, labels = extra_feats.to(device), labels.to(device)

            opt_r.zero_grad()
            preds = ranker(u_emb_batch, i_emb_batch, extra_feats=extra_feats)
            loss = rank_loss(preds, labels)
            loss.backward()
            opt_r.step()
            total_loss += loss.item()
        print(f"[Rank] Epoch {epoch+1}, Loss = {total_loss/len(rank_loader):.4f}")

    # -----------------------------
    # 11) Rank evaluation
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
            user_feat_matrix=all_user_feats.cpu(),
            item_feat_matrix=all_item_feats.cpu(),
        )
    print("Ranker Eval (Recall+Rank):", rank_results)


if __name__ == "__main__":
    main()