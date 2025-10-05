from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.feature_store import FeatureStore


@dataclass(frozen=True)
class RankDatasetConfig:
    num_negatives: int = 5
    max_history: int = 50
    candidate_neg_ratio: float = 0.6


class RankDataset(Dataset):
    """Provide (user, positive item, sampled negatives) tuples for ranker training."""

    def __init__(
        self,
        ratings_df,
        num_items,
        user_store: FeatureStore,
        item_store: FeatureStore,
        config: RankDatasetConfig = RankDatasetConfig(),
        positive_pairs: Sequence[Tuple[int, int]] | None = None,
        append_positive_to_history: bool | None = None,
        user_candidates: dict[int, Sequence[int]] | None = None,
    ):
        if ratings_df is None or ratings_df.empty:
            raise ValueError("ratings_df must contain training interactions")

        self.cfg = config
        self.num_items = int(num_items)
        self.user_store = user_store
        self.item_store = item_store
        self.user_candidates = user_candidates or {}
        self.rng = np.random.default_rng()
        self.catalog = np.arange(self.num_items, dtype=np.int64)

        self.user_rated: Dict[int, frozenset[int]] = {}
        ratings_sorted = ratings_df.sort_values(["user_idx", "timestamp"])
        for user, group in ratings_sorted.groupby("user_idx", sort=False):
            items = group["item_idx"].astype(int).tolist()
            self.user_rated[int(user)] = frozenset(items)

        if append_positive_to_history is None:
            self.append_positive_to_history = positive_pairs is None
        else:
            self.append_positive_to_history = bool(append_positive_to_history)

        self.base_examples = self._build_base_examples(
            ratings_sorted, positive_pairs
        )

    def _build_base_examples(
        self,
        ratings_sorted,
        positive_pairs: Sequence[Tuple[int, int]] | None,
    ) -> List[Tuple[int, int, List[int]]]:
        examples: List[Tuple[int, int, List[int]]] = []
        history_cache: Dict[int, List[int]] = defaultdict(list)

        if positive_pairs is None:
            for user, group in ratings_sorted.groupby("user_idx", sort=False):
                user_int = int(user)
                history: List[int] = []
                for row in group.itertuples():
                    item = int(row.item_idx)
                    examples.append((user_int, item, history[-self.cfg.max_history :]))
                    history.append(item)
                    history_cache[user_int] = history
        else:
            for raw_user, raw_item in positive_pairs:
                user_int = int(raw_user)
                item_int = int(raw_item)
                history = history_cache.get(user_int, [])
                examples.append((user_int, item_int, history[-self.cfg.max_history :]))
                if self.append_positive_to_history:
                    history.append(item_int)
                    history_cache[user_int] = history
                    self.user_rated[user_int] = frozenset(
                        set(self.user_rated.get(user_int, frozenset())) | {item_int}
                    )

        return examples

    def __len__(self) -> int:
        return len(self.base_examples)

    def __getitem__(self, idx: int):
        user_id, pos_item, history = self.base_examples[idx]
        history_tensor = torch.full(
            (self.cfg.max_history,), -1, dtype=torch.long
        )
        if history:
            trimmed = history[-self.cfg.max_history :]
            history_tensor[-len(trimmed) :] = torch.as_tensor(trimmed, dtype=torch.long)

        negatives = self._sample_negatives(user_id, pos_item)
        return {
            "user_id": torch.tensor(user_id, dtype=torch.long),
            "pos_item": torch.tensor(pos_item, dtype=torch.long),
            "neg_items": torch.as_tensor(negatives, dtype=torch.long),
            "hist_items": history_tensor,
            "hist_len": torch.tensor(len(history), dtype=torch.long),
        }

    @property
    def max_history(self) -> int:
        return self.cfg.max_history

    def _sample_negatives(self, user: int, positive_item: int) -> List[int]:
        if self.cfg.num_negatives <= 0 or self.num_items <= 0:
            return [positive_item]

        negatives: List[int] = []
        exclusion = set(self.user_rated.get(user, frozenset()))
        exclusion.add(int(positive_item))

        candidate_pool = self.user_candidates.get(user, [])
        candidate_quota = int(round(self.cfg.num_negatives * self.cfg.candidate_neg_ratio))
        candidate_quota = max(0, min(candidate_quota, self.cfg.num_negatives))

        if candidate_pool and candidate_quota > 0:
            filtered = [int(c) for c in candidate_pool if int(c) not in exclusion]
            if filtered:
                if len(filtered) <= candidate_quota:
                    chosen = filtered
                else:
                    chosen = self.rng.choice(filtered, size=candidate_quota, replace=False)
                negatives.extend(chosen.tolist() if isinstance(chosen, np.ndarray) else chosen)
                exclusion.update(negatives)

        remaining = self.cfg.num_negatives - len(negatives)
        if remaining <= 0:
            return negatives[: self.cfg.num_negatives]

        attempts = 0
        while remaining > 0 and attempts < 50:
            cand = int(self.rng.integers(0, self.num_items))
            if cand in exclusion:
                attempts += 1
                continue
            negatives.append(cand)
            exclusion.add(cand)
            remaining -= 1

        while len(negatives) < self.cfg.num_negatives:
            negatives.append(positive_item)

        return negatives[: self.cfg.num_negatives]

    def update_candidate_pool(self, new_candidates: dict[int, Sequence[int]]):
        self.user_candidates = new_candidates or {}
