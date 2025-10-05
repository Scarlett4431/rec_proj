from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.feature_store import FeatureStore


@dataclass
class NegativeConfig:
    easy: int = 3
    tail: int = 0
    hard: int = 0
    tail_alpha: float = 0.75
    tail_smoothing: float = 1.0
    hard_k: int = 0


class RecallDataset(Dataset):
    """Dataset for two-tower recall training with flexible negative sampling."""

    def __init__(
        self,
        ratings_df,
        user_store: FeatureStore,
        item_store: FeatureStore,
        num_items: Optional[int] = None,
        negative_cfg: NegativeConfig = NegativeConfig(),
        item_emb: Optional[torch.Tensor] = None,
    ):
        if ratings_df is None or ratings_df.empty:
            raise ValueError("ratings_df must contain training interactions")

        self.user_store = user_store
        self.item_store = item_store
        self.negative_cfg = negative_cfg
        self.pad_item_id = -1
        self._rng = np.random.default_rng()

        pairs = ratings_df[["user_idx", "item_idx"]].to_numpy(dtype=np.int64)
        self.users = torch.as_tensor(pairs[:, 0], dtype=torch.long)
        self.items = torch.as_tensor(pairs[:, 1], dtype=torch.long)

        self.num_items = (
            int(num_items)
            if num_items is not None
            else int(ratings_df["item_idx"].max()) + 1
        )

        self.user_rated: Dict[int, frozenset[int]] = {}
        for u, i in pairs:
            self.user_rated.setdefault(int(u), set()).add(int(i))
        self.user_rated = {u: frozenset(items) for u, items in self.user_rated.items()}

        self.user_consumed_np = {
            u: np.fromiter(items, dtype=np.int64)
            if items
            else np.empty(0, dtype=np.int64)
            for u, items in self.user_rated.items()
        }

        self.catalog = np.arange(self.num_items, dtype=np.int64)
        self._tail_probs = self._prepare_tail_weights(negative_cfg)
        self.hard_negatives = self._prepare_hard_negatives(
            negative_cfg, item_emb
        )

    def _prepare_tail_weights(self, cfg: NegativeConfig):
        if cfg.tail <= 0 or self.num_items <= 0:
            return None

        counts = np.full(self.num_items, cfg.tail_smoothing, dtype=np.float64)
        counts[: self.items.numel()] += np.bincount(
            self.items.cpu().numpy(), minlength=self.num_items
        )
        if cfg.tail_alpha > 0.0:
            weights = np.power(counts, -cfg.tail_alpha, where=counts > 0)
            weights[counts <= 0] = 0.0
        else:
            weights = np.ones_like(counts)
        total = weights.sum()
        if total <= 0:
            return None
        return weights / total

    def _prepare_hard_negatives(
        self, cfg: NegativeConfig, item_emb: Optional[torch.Tensor]
    ) -> Dict[int, np.ndarray]:
        if cfg.hard <= 0 or cfg.hard_k <= 0 or item_emb is None:
            return {}
        try:
            from src.faiss_index import FaissIndex

            index = FaissIndex(item_emb.detach().cpu())
            hard_map: Dict[int, np.ndarray] = {}
            unique_items = torch.unique(self.items).tolist()
            for item in unique_items:
                _, idxs = index.search(
                    item_emb[item].detach().cpu(), k=cfg.hard_k + 1
                )
                neighbors = [i for i in idxs.tolist() if i != item]
                hard_map[int(item)] = np.array(neighbors, dtype=np.int64)
            return hard_map
        except ImportError:
            print("[RecallDataset] FAISS not installed; skipping hard negatives")
            return {}

    def __len__(self) -> int:
        return self.users.numel()

    def __getitem__(self, idx: int):
        user_id = int(self.users[idx].item())
        pos_item = int(self.items[idx].item())
        consumed = self.user_rated.get(user_id, frozenset())

        sample = {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "hard_neg_items": self._sample_hard_negatives(user_id, pos_item, consumed),
            "easy_neg_items": self._sample_easy_negatives(user_id, pos_item, consumed),
            "tail_neg_items": self._sample_tail_negatives(user_id, pos_item, consumed),
        }
        return sample

    def _sample_hard_negatives(
        self, user: int, pos_item: int, consumed: Iterable[int]
    ) -> torch.Tensor:
        cfg = self.negative_cfg
        if cfg.hard <= 0:
            return torch.full((cfg.hard,), self.pad_item_id, dtype=torch.long)

        neighbors = self.hard_negatives.get(pos_item)
        if neighbors is None or neighbors.size == 0:
            return torch.full((cfg.hard,), self.pad_item_id, dtype=torch.long)

        eligible = [n for n in neighbors if n not in consumed and n != pos_item]
        if not eligible:
            return torch.full((cfg.hard,), self.pad_item_id, dtype=torch.long)

        take = cfg.hard
        if len(eligible) >= take:
            chosen = self._rng.choice(eligible, size=take, replace=False)
        else:
            chosen = self._rng.choice(eligible, size=take, replace=True)
        return torch.as_tensor(chosen, dtype=torch.long)

    def _sample_easy_negatives(
        self, user: int, pos_item: int, consumed: Iterable[int]
    ) -> torch.Tensor:
        cfg = self.negative_cfg
        if cfg.easy <= 0 or self.num_items <= 0:
            return torch.full((cfg.easy,), self.pad_item_id, dtype=torch.long)

        consumed_np = self.user_consumed_np.get(user, np.empty(0, dtype=np.int64))
        negatives: list[int] = []
        attempts = 0
        draw_size = max(cfg.easy * 4, cfg.easy + 8)
        while len(negatives) < cfg.easy and attempts < 3:
            draws = self._rng.integers(0, self.num_items, size=draw_size, dtype=np.int64)
            if consumed_np.size > 0:
                mask = ~np.isin(draws, consumed_np, assume_unique=False)
            else:
                mask = np.ones(draws.shape, dtype=bool)
            mask &= draws != pos_item
            filtered = draws[mask]
            if filtered.size > 0:
                take = min(cfg.easy - len(negatives), filtered.size)
                negatives.extend(filtered[:take].tolist())
            attempts += 1

        if len(negatives) < cfg.easy:
            remaining = cfg.easy - len(negatives)
            available = np.setdiff1d(self.catalog, consumed_np, assume_unique=True)
            available = available[available != pos_item]
            if available.size > 0:
                replace = available.size < remaining
                draws = self._rng.choice(available, size=remaining, replace=replace)
                negatives.extend(draws.tolist())

        if len(negatives) < cfg.easy:
            negatives.extend([self.pad_item_id] * (cfg.easy - len(negatives)))

        return torch.as_tensor(negatives[: cfg.easy], dtype=torch.long)

    def _sample_tail_negatives(
        self, user: int, pos_item: int, consumed: Iterable[int]
    ) -> torch.Tensor:
        cfg = self.negative_cfg
        if cfg.tail <= 0 or self._tail_probs is None:
            return torch.full((cfg.tail,), self.pad_item_id, dtype=torch.long)

        negatives: list[int] = []
        attempts = 0
        while len(negatives) < cfg.tail and attempts < 5:
            draw_count = max(cfg.tail * 4, cfg.tail + 8)
            draws = self._rng.choice(
                self.num_items,
                size=draw_count,
                replace=True,
                p=self._tail_probs,
            )
            for cand in draws:
                cand_int = int(cand)
                if cand_int == pos_item or cand_int in consumed:
                    continue
                if cand_int in negatives:
                    continue
                negatives.append(cand_int)
                if len(negatives) >= cfg.tail:
                    break
            attempts += 1

        if len(negatives) < cfg.tail:
            negatives.extend([self.pad_item_id] * (cfg.tail - len(negatives)))

        return torch.as_tensor(negatives[: cfg.tail], dtype=torch.long)

    def get_user_feature_matrix(self, max_user_id):
        return self.user_store.to_matrix(max_user_id)

    def get_item_feature_matrix(self, max_item_id):
        return self.item_store.to_matrix(max_item_id)
