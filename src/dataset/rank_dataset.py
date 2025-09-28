import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from src.features.feature_store import FeatureStore


class RankDataset(Dataset):
    """Ranking dataset built from recall candidates (two-stage pipeline)."""

    def __init__(self, ratings_df, num_items,
                 user_store: FeatureStore, item_store: FeatureStore,
                 num_negatives=5, user_emb=None, item_emb=None,
                 user_feat_tensor=None, item_feat_tensor=None,
                 user_candidates=None):

        self.num_items = num_items
        self.num_negatives = num_negatives
        self.user_emb = user_emb.float() if user_emb is not None else None
        self.item_emb = item_emb.float() if item_emb is not None else None

        # Feature stores (shared with recall dataset)
        self.user_store = user_store
        self.item_store = item_store

        # Precompute dense feature matrices for fast indexing
        if user_feat_tensor is not None:
            self.user_feat_tensor = user_feat_tensor.float()
        else:
            max_user_id = max(user_store.data.keys()) if user_store.data else -1
            self.user_feat_tensor = user_store.to_matrix(max_user_id).float()

        if item_feat_tensor is not None:
            self.item_feat_tensor = item_feat_tensor.float()
        else:
            max_item_id = max(item_store.data.keys()) if item_store.data else -1
            self.item_feat_tensor = item_store.to_matrix(max_item_id).float()

        # Track interactions
        self.user_item_pairs = ratings_df[["user_idx", "item_idx"]].values
        self.user_rated = defaultdict(set)
        for u, i in self.user_item_pairs:
            self.user_rated[u].add(i)

        # Recall candidate pool per user (FAISS shortlist)
        self.user_candidates = user_candidates or {}

        # Positive + negative samples
        samples = []
        rng = np.random.default_rng()
        all_items = np.arange(num_items)
        for u, i in self.user_item_pairs:
            samples.append((u, i, 1))

            if num_negatives <= 0:
                continue

            rated = self.user_rated[u]
            candidate_pool = self.user_candidates.get(u)
            negatives_source = None
            if candidate_pool:
                negatives_source = np.array(candidate_pool, dtype=np.int64)

            needed = num_negatives
            attempts = 0
            while needed > 0:
                if negatives_source is not None:
                    draws = rng.choice(negatives_source, size=max(needed, 1), replace=True)
                else:
                    draws = rng.choice(all_items, size=max(needed, 1), replace=True)

                candidates = [item for item in draws if item not in rated]

                if not candidates:
                    attempts += 1
                    if attempts > 10:
                        break
                    continue

                take = candidates[:needed]
                samples.extend((u, j, 0) for j in take)
                needed -= len(take)

        samples_np = np.array(samples, dtype=np.int64)
        self.samples = torch.from_numpy(samples_np)
        self.labels = self.samples[:, 2].float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        u = sample[0].item()
        i = sample[1].item()

        # Recall embeddings (frozen from two-tower)
        u_emb = self.user_emb[u]
        i_emb = self.item_emb[i]

        # Feature tensors (not concatenated)
        u_feats = self.user_feat_tensor[u]
        i_feats = self.item_feat_tensor[i]

        return {
            "u_emb": u_emb,
            "i_emb": i_emb,
            "u_feats": u_feats,
            "i_feats": i_feats,
            "label": self.labels[idx],
        }
