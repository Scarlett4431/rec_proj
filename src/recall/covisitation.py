from collections import defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Optional

import pandas as pd


def _group_user_items(ratings: pd.DataFrame) -> pd.Series:
    return (
        ratings.sort_values(["user_idx", "timestamp"], ascending=[True, True])
        .groupby("user_idx")
        ["item_idx"]
        .apply(list)
    )


def build_covisitation_index(
    ratings: pd.DataFrame,
    max_items_per_user: int = 50,
    top_k: int = 200,
) -> Dict[int, List[tuple]]:
    """Build symmetric item→neighbors map weighted by co-view counts."""

    user_items = _group_user_items(ratings)
    co_counts: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(float))

    for items in user_items:
        recent = items[-max_items_per_user:]
        if len(recent) < 2:
            continue
        for a, b in combinations(recent, 2):
            if a == b:
                continue
            co_counts[a][b] += 1.0
            co_counts[b][a] += 1.0

    covis_index: Dict[int, List[tuple]] = {}
    for item, neighbors in co_counts.items():
        if not neighbors:
            continue
        sorted_neighbors = sorted(neighbors.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        covis_index[item] = sorted_neighbors
    return covis_index


def build_user_covisitation_candidates(
    ratings: pd.DataFrame,
    covis_index: Dict[int, List[tuple]],
    top_k: int = 100,
    max_history: int = 25,
    user_consumed: Optional[Dict[int, Iterable[int]]] = None,
) -> Dict[int, List[int]]:
    """Aggregate co-vis neighbors per user, filtering consumed items."""

    user_items = _group_user_items(ratings)
    user_candidates: Dict[int, List[int]] = {}

    for user, items in user_items.items():
        recent = items[-max_history:]
        seen = set(recent)
        if user_consumed is not None:
            seen.update(user_consumed.get(user, ()))
        scores: Dict[int, float] = defaultdict(float)
        for item in recent:
            for neighbor, score in covis_index.get(item, []):
                if neighbor in seen:
                    continue
                scores[neighbor] += score

        if not scores:
            user_candidates[user] = []
            continue

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        user_candidates[user] = [item for item, _ in ranked]

    return user_candidates
