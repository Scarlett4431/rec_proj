from typing import Dict, Iterable, List, Optional

import pandas as pd


def build_popular_items(ratings: pd.DataFrame, top_k: int = 200) -> List[int]:
    """Return globally most-popular item indices by interaction count."""
    counts = (
        ratings["item_idx"].value_counts()
        .sort_values(ascending=False)
        .head(top_k)
    )
    return counts.index.tolist()


def build_user_popularity_candidates(
    ratings: pd.DataFrame,
    popular_items: List[int],
    num_users: int,
    top_k: int = 100,
    user_consumed: Optional[Dict[int, Iterable[int]]] = None,
) -> Dict[int, List[int]]:
    """Recommend top popular items per user, filtering out consumed ones."""

    user_history = (
        ratings.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )

    user_candidates: Dict[int, List[int]] = {}
    for user in range(num_users):
        seen = set(user_history.get(user, set()))
        if user_consumed is not None:
            seen.update(user_consumed.get(user, ()))
        filtered = [item for item in popular_items if item not in seen]
        user_candidates[user] = filtered[:top_k]

    return user_candidates
