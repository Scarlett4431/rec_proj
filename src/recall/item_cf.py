import math
from collections import defaultdict
from typing import Dict, Iterable, List

import pandas as pd


def _user_item_sequences(ratings: pd.DataFrame) -> Iterable[List[int]]:
    return (
        ratings.sort_values(["user_idx", "timestamp"], ascending=[True, True])
        .groupby("user_idx")
        ["item_idx"]
        .apply(list)
    )


def build_item_cf_index(
    ratings: pd.DataFrame,
    max_items_per_user: int = 200,
    top_k: int = 200,
) -> Dict[int, List[tuple]]:
    """Item-based CF similarity index using normalized co-occurrence."""

    user_items = _user_item_sequences(ratings)
    item_popularity = ratings["item_idx"].value_counts().to_dict()
    pair_counts: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for items in user_items:
        if not items:
            continue
        recent = items[-max_items_per_user:]
        unique_items = list(dict.fromkeys(recent))
        if len(unique_items) < 2:
            continue
        weight = 1.0 / math.log1p(len(unique_items))
        for i in range(len(unique_items)):
            a = unique_items[i]
            for j in range(i + 1, len(unique_items)):
                b = unique_items[j]
                pair_counts[a][b] += weight
                pair_counts[b][a] += weight

    index: Dict[int, List[tuple]] = {}
    for item, neighbors in pair_counts.items():
        pop_a = math.sqrt(item_popularity.get(item, 1.0))
        scored = []
        for neighbor, count in neighbors.items():
            pop_b = math.sqrt(item_popularity.get(neighbor, 1.0))
            norm = pop_a * pop_b
            score = count / (norm + 1e-6)
            scored.append((neighbor, score))

        scored.sort(key=lambda kv: kv[1], reverse=True)
        index[item] = scored[:top_k]

    return index


def build_user_itemcf_candidates(
    ratings: pd.DataFrame,
    item_cf_index: Dict[int, List[tuple]],
    top_k: int = 100,
    max_history: int = 50,
) -> Dict[int, List[int]]:
    """Aggregate item-CF neighbors for each user."""

    user_items = _user_item_sequences(ratings)
    user_candidates: Dict[int, List[int]] = {}

    for user, items in user_items.items():
        recent = items[-max_history:]
        seen = set(recent)
        scores: Dict[int, float] = defaultdict(float)
        for item in recent:
            for neighbor, score in item_cf_index.get(item, []):
                if neighbor in seen:
                    continue
                scores[neighbor] += score

        if not scores:
            user_candidates[user] = []
            continue

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        user_candidates[user] = [item for item, _ in ranked]

    return user_candidates

