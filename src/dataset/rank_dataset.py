from collections import defaultdict
from typing import Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.features.feature_store import FeatureStore


class RankDataset(Dataset):
    """Ranking dataset with random negatives and configurable positives."""

    def __init__(
        self,
        ratings_df,
        num_items,
        user_store: FeatureStore,
        item_store: FeatureStore,
        num_negatives: int = 5,
        user_emb=None,
        item_emb=None,
        user_feat_tensor=None,
        item_feat_tensor=None,
        max_history: int = 50,
        positive_pairs: Sequence[Tuple[int, int]] | None = None,
        append_positive_to_history: bool | None = None,
    ):
        if user_emb is None or item_emb is None:
            raise ValueError("RankDataset requires precomputed user and item embeddings")

        self.num_items = int(num_items)
        self.num_negatives = max(int(num_negatives), 0)
        self.max_history = int(max_history)
        self.user_emb = user_emb.float()
        self.item_emb = item_emb.float()

        self.user_store = user_store
        self.item_store = item_store

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

        # Build history/consumption maps from ratings_df (training interactions)
        self.user_histories: dict[int, list[int]] = {}
        self.user_rated: defaultdict[int, set[int]] = defaultdict(set)
        ratings_sorted = None
        if ratings_df is not None and not ratings_df.empty:
            ratings_sorted = ratings_df.sort_values(["user_idx", "timestamp"])
            for user, group in ratings_sorted.groupby("user_idx", sort=False):
                user_int = int(user)
                history_items = group["item_idx"].astype(int).tolist()
                self.user_histories[user_int] = history_items
                self.user_rated[user_int].update(history_items)

        if append_positive_to_history is None:
            self.append_positive_to_history = positive_pairs is None
        else:
            self.append_positive_to_history = bool(append_positive_to_history)

        samples: list[tuple[int, int, int]] = []
        histories: list[list[int]] = []
        history_lengths: list[int] = []
        rng = np.random.default_rng()
        all_items = (
            np.arange(self.num_items, dtype=np.int64)
            if self.num_items > 0
            else np.empty(0, dtype=np.int64)
        )

        def sample_negatives(user_id: int, positive_item: int, history_snapshot: list[int]):
            if self.num_negatives <= 0 or self.num_items <= 0:
                return
            exclusion = set(self.user_rated.get(user_id, set()))
            exclusion.add(int(positive_item))

            unique_budget = self.num_items - len(exclusion)
            if unique_budget <= 0:
                return

            needed = self.num_negatives
            attempts = 0
            while needed > 0 and attempts < 10:
                draw_size = max(needed * 2, needed + 8)
                draws = rng.choice(all_items, size=draw_size, replace=True)
                negatives: list[int] = []
                for cand in draws:
                    cand_int = int(cand)
                    if cand_int in exclusion:
                        continue
                    exclusion.add(cand_int)
                    negatives.append(cand_int)
                    if len(negatives) >= needed:
                        break

                if not negatives:
                    attempts += 1
                    continue

                take = negatives[:needed]
                for neg in take:
                    samples.append((user_id, neg, 0))
                    histories.append(history_snapshot.copy())
                    history_lengths.append(len(history_snapshot))
                needed -= len(take)

        if positive_pairs is None:
            if ratings_sorted is None:
                ratings_iter = []
            else:
                ratings_iter = ratings_sorted.groupby("user_idx", sort=False)

            for user, group in ratings_iter:
                user_int = int(user)
                history: list[int] = []
                for row in group.itertuples():
                    item = int(row.item_idx)
                    hist_trim = history[-self.max_history :]
                    history_snapshot = hist_trim.copy()

                    samples.append((user_int, item, 1))
                    histories.append(history_snapshot)
                    history_lengths.append(len(history_snapshot))

                    sample_negatives(user_int, item, history_snapshot)

                    history.append(item)
        else:
            pairs = [(int(u), int(i)) for u, i in positive_pairs]
            history_buffer = {
                user: list(self.user_histories.get(user, [])) for user, _ in pairs
            }
            for user_int, item_int in pairs:
                base_history = history_buffer.get(user_int)
                if base_history is None:
                    base_history = []
                    history_buffer[user_int] = base_history

                hist_trim = base_history[-self.max_history :]
                history_snapshot = hist_trim.copy()

                samples.append((user_int, item_int, 1))
                histories.append(history_snapshot)
                history_lengths.append(len(history_snapshot))

                sample_negatives(user_int, item_int, history_snapshot)

                if self.append_positive_to_history:
                    base_history.append(item_int)
                    self.user_rated[user_int].add(item_int)

        if samples:
            samples_np = np.array(samples, dtype=np.int64)
        else:
            samples_np = np.zeros((0, 3), dtype=np.int64)
        self.samples = torch.from_numpy(samples_np)
        self.labels = (
            self.samples[:, 2].float()
            if self.samples.numel()
            else torch.zeros((0,), dtype=torch.float32)
        )

        history_pad = torch.full((len(histories), self.max_history), -1, dtype=torch.long)
        for idx, hist in enumerate(histories):
            if not hist:
                continue
            trimmed = hist[-self.max_history :]
            if trimmed:
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

        u_emb = self.user_emb[u]
        i_emb = self.item_emb[i]
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
