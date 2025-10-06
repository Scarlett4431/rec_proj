from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


def _normalize_genres(movies_df: pd.DataFrame) -> pd.DataFrame:
    movies = movies_df.copy()
    if "genres" not in movies.columns:
        movies["genres"] = ""
    movies["genres"] = (
        movies["genres"].fillna("")
        .apply(lambda s: [g for g in str(s).split("|") if g and g != "(no genres listed)"])
    )
    movies["genres_list"] = movies["genres"]
    return movies


def build_genre_sources(
    train_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_consumed: Dict[int, Sequence[int]],
    genre_top_k: int = 3,
    genre_item_limit: int = 200,
) -> Tuple[Dict[str, List[int]], Dict[int, List[int]]]:
    """Return genre → items map and per-user genre candidate list."""

    if movies_df is None or movies_df.empty:
        return {}, {}

    movies = _normalize_genres(movies_df)
    if movies.empty:
        return {}, {}

    item_genres = movies.set_index("item_idx")["genres_list"].to_dict()
    item_popularity = train_df["item_idx"].value_counts()

    genre_item_map: Dict[str, List[int]] = defaultdict(list)
    for item_id, genres in item_genres.items():
        if not isinstance(genres, list):
            continue
        pop = item_popularity.get(item_id, 0)
        for genre in genres:
            genre_item_map[genre].append((pop, int(item_id)))

    for genre, pairs in genre_item_map.items():
        pairs.sort(key=lambda x: (-x[0], x[1]))
        genre_item_map[genre] = [item for _, item in pairs[:genre_item_limit]]

    merged = train_df.merge(
        movies[["item_idx", "genres_list"]],
        on="item_idx",
        how="left",
    )
    exploded = merged.explode("genres_list").dropna(subset=["genres_list"])
    if exploded.empty:
        return genre_item_map, {}

    genre_counts = (
        exploded.groupby(["user_idx", "genres_list"])["item_idx"].count().reset_index(name="count")
    )
    genre_counts = genre_counts.sort_values(
        ["user_idx", "count", "genres_list"], ascending=[True, False, True]
    )
    user_top_genres = genre_counts.groupby("user_idx")["genres_list"].apply(
        lambda s: s.head(genre_top_k).tolist()
    )

    user_genre_candidates: Dict[int, List[int]] = {}
    for user, genres in user_top_genres.items():
        consumed = set(user_consumed.get(int(user), []))
        seen: set[int] = set()
        candidates: List[int] = []
        for genre in genres:
            for item_id in genre_item_map.get(genre, []):
                if item_id in consumed or item_id in seen:
                    continue
                candidates.append(item_id)
                seen.add(item_id)
                if len(candidates) >= genre_item_limit:
                    break
            if len(candidates) >= genre_item_limit:
                break
        if candidates:
            user_genre_candidates[int(user)] = candidates

    return dict(genre_item_map), user_genre_candidates

