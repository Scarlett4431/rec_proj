import torch
from torch.utils.data import Dataset
import numpy as np
from collections import defaultdict
from typing import List
from src.features.feature_store import FeatureStore


class RankDataset(Dataset):
    """Ranking dataset built from recall candidates (two-stage pipeline)."""

    def __init__(
        self,
        ratings_df,
        num_items,
        user_store: FeatureStore,
        item_store: FeatureStore,
        num_negatives=5,
        user_emb=None,
        item_emb=None,
        user_feat_tensor=None,
        item_feat_tensor=None,
        user_candidates=None,
        max_history: int = 50,
    ):

        self.num_items = num_items
        self.num_negatives = num_negatives
        self.max_history = max_history
        self.user_emb = user_emb.float() if user_emb is not None else None
        self.item_emb = item_emb.float() if item_emb is not None else None
        self.max_history = max_history

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
        ratings_sorted = ratings_df.sort_values(["user_idx", "timestamp"])
        self.user_item_pairs = ratings_sorted[["user_idx", "item_idx"]].values
        self.user_rated = defaultdict(set)
        for u, i in self.user_item_pairs:
            self.user_rated[u].add(i)

        # Recall candidate pool per user (FAISS shortlist)
        self.user_candidates = user_candidates or {}

        # Positive + negative samples
        samples = []
        histories = []
        history_lengths = []
        rng = np.random.default_rng()
        all_items = np.arange(num_items)
        for user, group in ratings_sorted.groupby("user_idx", sort=False):
            history = []
            rated = self.user_rated[user]
            candidate_pool = self.user_candidates.get(user)
            negatives_source = (np.array(candidate_pool, dtype=np.int64) if candidate_pool else None)

            for row in group.itertuples():
                item = int(row.item_idx)
                hist_trim = history[-self.max_history :]
                samples.append((user, item, 1))
                histories.append(hist_trim.copy())
                history_lengths.append(len(hist_trim))

                if num_negatives > 0:
                    needed = num_negatives
                    attempts = 0
                    while needed > 0:
                        if negatives_source is not None:
                            draws = rng.choice(negatives_source, size=max(needed, 1), replace=True)
                        else:
                            draws = rng.choice(all_items, size=max(needed, 1), replace=True)

                        candidates = [cand for cand in draws if cand not in rated]
                        if not candidates:
                            attempts += 1
                            if attempts > 10:
                                break
                            continue

                        take = candidates[:needed]
                        for neg in take:
                            samples.append((user, int(neg), 0))
                            histories.append(hist_trim.copy())
                            history_lengths.append(len(hist_trim))
                        needed -= len(take)

                history.append(item)

        samples_np = np.array(samples, dtype=np.int64)
        self.samples = torch.from_numpy(samples_np)
        self.labels = self.samples[:, 2].float()

        history_pad = torch.full((len(histories), self.max_history), -1, dtype=torch.long)
        for idx, hist in enumerate(histories):
            if not hist:
                continue
            trimmed = hist[-self.max_history :]
            hist_tensor = torch.tensor(trimmed, dtype=torch.long)
            history_pad[idx, -len(trimmed) :] = hist_tensor
        self.history_items = history_pad
        self.history_lengths = torch.tensor(history_lengths, dtype=torch.long)

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

        hist_items = self.history_items[idx]
        hist_len = self.history_lengths[idx]

        return {
            "u_emb": u_emb,
            "i_emb": i_emb,
            "u_feats": u_feats,
            "i_feats": i_feats,
            "label": self.labels[idx],
            "hist_items": hist_items,
            "hist_len": hist_len,
        }
