import torch
from torch.utils.data import Dataset
import numpy as np
from src.features.feature_store import FeatureStore


class RecallDataset(Dataset):
    """
    Dataset for recall training (in-batch negatives + optional hard negatives).
    Uses generalized FeatureStore: numeric + categorical + bucketized features.
    """

    def __init__(self, ratings_df, user_store: FeatureStore, item_store: FeatureStore,
                 item_emb=None, hard_neg_k=0, hard_neg_samples=0):
        """
        ratings_df: DataFrame with [user_idx, item_idx]
        user_store, item_store: pre-built FeatureStore instances
        item_emb: torch.Tensor [num_items, d], pretrained embeddings (for mining)
        hard_neg_k: if >0, use FAISS to mine top-k neighbors as hard negative pool
        hard_neg_samples: how many to sample per positive
        """
        self.pairs = ratings_df[["user_idx", "item_idx"]].values

        # Feature stores are shared (avoid rebuilding + ensure consistent columns)
        self.user_store = user_store
        self.item_store = item_store

        self.user_feat_dim = self.user_store.total_dim
        self.item_feat_dim = self.item_store.total_dim

        # Track user → items they consumed
        user_rated = {}
        for u, i in self.pairs:
            user_rated.setdefault(u, set()).add(i)
        self.user_rated = {u: frozenset(items) for u, items in user_rated.items()}

        self.pad_item_id = -1
        self.hard_neg_samples = hard_neg_samples
        self.hard_negatives = {}

        # Precompute item neighbors globally if requested
        if hard_neg_k > 0 and item_emb is not None:
            try:
                from src.faiss_index import FaissIndex
                index = FaissIndex(item_emb.detach().cpu())
                unique_pos_items = np.unique(self.pairs[:, 1])
                for pos_item in unique_pos_items:
                    _, idxs = index.search(item_emb[pos_item].detach().cpu(),
                                           k=hard_neg_k + 1)
                    neighbors = [i for i in idxs.tolist() if i != pos_item]
                    if neighbors:
                        self.hard_negatives[pos_item] = np.array(neighbors, dtype=np.int64)
            except ImportError:
                print("[RecallDataset] FAISS not installed, skipping hard negatives")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        u, pos_item = self.pairs[idx]

        # Features: tensors (from FeatureStore)
        u_feats = self.user_store.get_tensor(u)
        i_feats = self.item_store.get_tensor(pos_item)

        sample = {
            "user_id": torch.tensor(u, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "user_feats": u_feats,         # tensor
            "pos_item_feats": i_feats,     # tensor
        }

        # Hard negatives
        if self.hard_neg_samples > 0 and pos_item in self.hard_negatives:
            candidates = self.hard_negatives[pos_item]
            consumed = self.user_rated.get(u, frozenset())
            filtered = np.array([c for c in candidates if c not in consumed], dtype=np.int64)

            if filtered.size == 0:
                neg_items = torch.full((self.hard_neg_samples,), self.pad_item_id, dtype=torch.long)
            else:
                replace = filtered.size < self.hard_neg_samples
                sampled = np.random.choice(filtered, size=self.hard_neg_samples, replace=replace)
                neg_items = torch.from_numpy(sampled.astype(np.int64))
            sample["hard_neg_items"] = neg_items
        else:
            sample["hard_neg_items"] = torch.full((self.hard_neg_samples,), self.pad_item_id, dtype=torch.long)

        return sample

    def get_user_feature_matrix(self, max_user_id):
        return self.user_store.to_matrix(max_user_id)

    def get_item_feature_matrix(self, max_item_id):
        return self.item_store.to_matrix(max_item_id)
