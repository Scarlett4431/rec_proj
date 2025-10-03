from typing import Dict, List

import torch
from torch.utils.data import Dataset
import numpy as np
from src.features.feature_store import FeatureStore


class RecallDataset(Dataset):
    """
    Dataset for recall training (in-batch negatives + optional hard negatives).
    Uses generalized FeatureStore: numeric + categorical + bucketized features.
    """

    def __init__(
        self,
        ratings_df,
        user_store: FeatureStore,
        item_store: FeatureStore,
        item_emb=None,
        hard_neg_k=0,
        hard_neg_samples=0,
        easy_neg_samples=0,
        tail_neg_samples=0,
        tail_sampling_alpha: float = 0.75,
        tail_sampling_smoothing: float = 1.0,
        num_items=None,
    ):
        """
        ratings_df: DataFrame with [user_idx, item_idx]
        user_store, item_store: pre-built FeatureStore instances
        item_emb: torch.Tensor [num_items, d], pretrained embeddings (for mining)
        hard_neg_k: if >0, use FAISS to mine top-k neighbors as hard negative pool
        hard_neg_samples: how many to sample per positive
        easy_neg_samples: number of random easy negatives per positive
        tail_neg_samples: number of popularity-balanced negatives per positive
        tail_sampling_alpha: exponent for inverse-popularity sampling (higher -> more tail)
        tail_sampling_smoothing: additive smoothing before inverse weighting
        num_items: total catalog size (required when easy_neg_samples > 0; inferred if None)
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
        self.user_consumed_np = {
            u: np.fromiter(items, dtype=np.int64, count=len(items)) if items else np.empty(0, dtype=np.int64)
            for u, items in self.user_rated.items()
        }

        self.pad_item_id = -1
        self.hard_neg_samples = int(max(hard_neg_samples, 0))
        self.easy_neg_samples = int(max(easy_neg_samples, 0))
        self.tail_neg_samples = int(max(tail_neg_samples, 0))
        self.tail_sampling_alpha = float(max(tail_sampling_alpha, 0.0))
        self.tail_sampling_smoothing = float(max(tail_sampling_smoothing, 0.0))
        self.hard_negatives = {}
        if num_items is not None:
            self.num_items = int(num_items)
        elif len(self.pairs) > 0:
            self.num_items = int(self.pairs[:, 1].max()) + 1
        else:
            self.num_items = 0
        self._rng = np.random.default_rng()
        self.user_available_cache: Dict[int, np.ndarray] = {}
        self._catalog = np.arange(self.num_items, dtype=np.int64) if self.num_items > 0 else np.empty(0, dtype=np.int64)
        self._tail_sampling_probs = None
        if self.tail_neg_samples > 0 and self.num_items > 0:
            counts = np.full(self.num_items, self.tail_sampling_smoothing, dtype=np.float64)
            if len(self.pairs) > 0:
                item_counts = np.bincount(self.pairs[:, 1], minlength=self.num_items).astype(np.float64)
                counts[: item_counts.shape[0]] += item_counts
            if self.tail_sampling_alpha > 0.0:
                weights = np.power(counts, -self.tail_sampling_alpha, where=counts > 0, out=np.zeros_like(counts))
                weights[counts <= 0] = 0.0
            else:
                weights = np.ones_like(counts)
            total = weights.sum()
            if total > 0:
                self._tail_sampling_probs = weights / total
            elif self.num_items > 0:
                self._tail_sampling_probs = np.full(self.num_items, 1.0 / self.num_items, dtype=np.float64)


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

        sample = {
            "user_id": torch.tensor(u, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
        }

        consumed = self.user_rated.get(u, frozenset())
        consumed_set = set(consumed)
        consumed_np = self.user_consumed_np.get(u, np.empty(0, dtype=np.int64))

        # Hard negatives
        if self.hard_neg_samples > 0 and pos_item in self.hard_negatives:
            candidates = self.hard_negatives[pos_item]
            filtered = np.array([c for c in candidates if c not in consumed_set and c != pos_item], dtype=np.int64)

            if filtered.size == 0:
                neg_items = torch.full((self.hard_neg_samples,), self.pad_item_id, dtype=torch.long)
            else:
                replace = filtered.size < self.hard_neg_samples
                sampled = np.random.choice(filtered, size=self.hard_neg_samples, replace=replace)
                neg_items = torch.from_numpy(sampled.astype(np.int64))
            sample["hard_neg_items"] = neg_items
        else:
            sample["hard_neg_items"] = torch.full((self.hard_neg_samples,), self.pad_item_id, dtype=torch.long)

        if self.easy_neg_samples > 0 and self.num_items > 0:
            easy_negs: List[int] = []
            attempts = 0
            draw_size = max(self.easy_neg_samples * 4, self.easy_neg_samples + 8)
            while len(easy_negs) < self.easy_neg_samples and attempts < 3:
                draws = self._rng.integers(0, self.num_items, size=draw_size, dtype=np.int64)
                if consumed_np.size > 0:
                    mask = ~np.isin(draws, consumed_np, assume_unique=False)
                else:
                    mask = np.ones(draws.shape, dtype=bool)
                mask &= draws != pos_item
                filtered = draws[mask]
                if filtered.size > 0:
                    take = min(self.easy_neg_samples - len(easy_negs), filtered.size)
                    easy_negs.extend(filtered[:take].tolist())
                attempts += 1

            if len(easy_negs) < self.easy_neg_samples and len(consumed_set) < self.num_items - 1:
                remaining = self.easy_neg_samples - len(easy_negs)
                available = self.user_available_cache.get(u)
                if available is None:
                    available = np.setdiff1d(self._catalog, consumed_np, assume_unique=True)
                    if available.size > 0 and pos_item not in consumed_set:
                        available = available[available != pos_item]
                    self.user_available_cache[u] = available
                if available.size > 0:
                    replace = available.size < remaining
                    draws = self._rng.choice(available, size=remaining, replace=replace)
                    easy_negs.extend(draws.tolist())

            if len(easy_negs) < self.easy_neg_samples:
                easy_negs.extend([self.pad_item_id] * (self.easy_neg_samples - len(easy_negs)))

            sample["easy_neg_items"] = torch.tensor(easy_negs[:self.easy_neg_samples], dtype=torch.long)
        else:
            sample["easy_neg_items"] = torch.full((self.easy_neg_samples,), self.pad_item_id, dtype=torch.long)

        if self.tail_neg_samples > 0 and self._tail_sampling_probs is not None:
            tail_negs: List[int] = []
            attempts = 0
            desired = self.tail_neg_samples
            while len(tail_negs) < desired and attempts < 5:
                draws = self._rng.choice(
                    self.num_items,
                    size=max(desired * 4, desired + 8),
                    replace=True,
                    p=self._tail_sampling_probs,
                )
                for cand in draws:
                    cand_int = int(cand)
                    if cand_int == pos_item or cand_int in consumed_set:
                        continue
                    if cand_int in tail_negs:
                        continue
                    tail_negs.append(cand_int)
                    if len(tail_negs) >= desired:
                        break
                attempts += 1

            if len(tail_negs) < desired:
                tail_negs.extend([self.pad_item_id] * (desired - len(tail_negs)))

            sample["tail_neg_items"] = torch.tensor(tail_negs[:desired], dtype=torch.long)
        else:
            sample["tail_neg_items"] = torch.full((self.tail_neg_samples,), self.pad_item_id, dtype=torch.long)

        return sample

    def get_user_feature_matrix(self, max_user_id):
        return self.user_store.to_matrix(max_user_id)

    def get_item_feature_matrix(self, max_item_id):
        return self.item_store.to_matrix(max_item_id)
