from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import pandas as pd

from src.recall.covisitation import build_covisitation_index, build_user_covisitation_candidates
from src.recall.item_cf import build_item_cf_index, build_user_itemcf_candidates
from src.recall.popularity import build_popular_items, build_user_popularity_candidates
from src.recall.hybrid import merge_candidate_lists
from src.faiss_index import FaissIndex


@dataclass
class CandidateSources:
    covis_index: Dict[int, list]
    item_cf_index: Dict[int, list]
    popular_items: Iterable[int]
    covis_user_candidates: Dict[int, list]
    itemcf_user_candidates: Dict[int, list]
    popular_user_candidates: Dict[int, list]
    genre_item_map: Dict[str, list]
    user_top_genres: Dict[int, list]


def build_candidate_sources(
    train_df,
    movies_df: pd.DataFrame,
    num_users: int,
    user_consumed,
    covis_k: int = 150,
    itemcf_k: int = 150,
    popular_k: int = 100,
    genre_top_k: int = 3,
    genre_item_limit: int = 100,
) -> CandidateSources:
    covis_index = build_covisitation_index(train_df, max_items_per_user=50, top_k=200)
    item_cf_index = build_item_cf_index(train_df, max_items_per_user=100, top_k=200)
    popular_items = build_popular_items(train_df, top_k=500)

    covis_user_candidates = build_user_covisitation_candidates(
        train_df,
        covis_index,
        top_k=covis_k,
        max_history=25,
        user_consumed=user_consumed,
    )
    itemcf_user_candidates = build_user_itemcf_candidates(
        train_df,
        item_cf_index,
        top_k=itemcf_k,
        max_history=50,
        user_consumed=user_consumed,
    )
    popular_user_candidates = build_user_popularity_candidates(
        train_df,
        popular_items,
        num_users=num_users,
        top_k=popular_k,
        user_consumed=user_consumed,
    )

    genre_item_map, user_top_genres = _build_genre_sources(
        train_df,
        movies_df,
        user_consumed,
        genre_top_k=genre_top_k,
        genre_item_limit=genre_item_limit,
    )

    return CandidateSources(
        covis_index=covis_index,
        item_cf_index=item_cf_index,
        popular_items=popular_items,
        covis_user_candidates=covis_user_candidates,
        itemcf_user_candidates=itemcf_user_candidates,
        popular_user_candidates=popular_user_candidates,
        genre_item_map=genre_item_map,
        user_top_genres=user_top_genres,
    )


def build_hybrid_candidates(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_consumed,
    sources: CandidateSources,
    candidate_k: int = 100,
    faiss_weight: float = 0.8,
    covis_weight: float = 0,
    itemcf_weight: float = 0.4,
    popular_weight: float = 0.1,
    genre_weight: float = 0.15,
    genre_boost_limit: int = 20,
) -> Dict[int, list]:
    faiss_index = FaissIndex(item_emb.detach().cpu())
    num_users = user_emb.size(0)
    num_items = item_emb.size(0)
    user_candidates = {}

    for u in range(num_users):
        consumed = user_consumed.get(u, set())
        if isinstance(consumed, list):
            consumed = set(consumed)
        search_k = min(num_items, candidate_k + len(consumed)) if consumed else candidate_k
        _, idxs = faiss_index.search(user_emb[u].detach().cpu(), k=search_k)
        faiss_filtered = [int(i) for i in idxs.tolist() if i >= 0 and i not in consumed]
        faiss_candidates = faiss_filtered[:candidate_k]

        covis_list = sources.covis_user_candidates.get(u, [])
        itemcf_list = sources.itemcf_user_candidates.get(u, [])
        popular_list = sources.popular_user_candidates.get(u, [])

        genre_boost = []
        if genre_weight > 0 and sources.user_top_genres:
            seen = set()
            for genre in sources.user_top_genres.get(u, []):
                for item_id in sources.genre_item_map.get(genre, []):
                    if item_id in consumed or item_id in seen:
                        continue
                    genre_boost.append(item_id)
                    seen.add(item_id)
                    if len(genre_boost) >= genre_boost_limit:
                        break
                if len(genre_boost) >= genre_boost_limit:
                    break

        user_candidates[u] = merge_candidate_lists(
            [
                (faiss_candidates, faiss_weight),
                (covis_list, covis_weight),
                (itemcf_list, itemcf_weight),
                (popular_list, popular_weight),
                (genre_boost, genre_weight),
            ],
            candidate_k,
        )

    return user_candidates


def _build_genre_sources(
    train_df,
    movies_df: pd.DataFrame,
    user_consumed,
    genre_top_k: int,
    genre_item_limit: int,
):
    if movies_df is None or movies_df.empty:
        return {}, {}

    movies = movies_df.copy()
    genres_series = movies.get("genres")
    if genres_series is None:
        return {}, {}

    if genres_series.dtype != object or not isinstance(genres_series.iloc[0], list):
        genres_series = genres_series.fillna("").apply(
            lambda s: [g for g in str(s).split("|") if g and g != "(no genres listed)"]
        )
    movies["genres_list"] = genres_series

    item_genres = movies.set_index("item_idx")["genres_list"].to_dict()

    # Build genre -> items sorted by popularity within training data
    item_popularity = train_df["item_idx"].value_counts()
    genre_item_map: Dict[str, list] = {}
    for item_id, genres in item_genres.items():
        if not isinstance(genres, list):
            continue
        pop = item_popularity.get(item_id, 0)
        for genre in genres:
            genre_item_map.setdefault(genre, []).append((pop, item_id))

    for genre, pairs in genre_item_map.items():
        pairs.sort(key=lambda tup: (-tup[0], tup[1]))
        genre_item_map[genre] = [item for _, item in pairs[:genre_item_limit]]

    # Build per-user top genres
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
    genre_counts = genre_counts.sort_values(["user_idx", "count", "genres_list"], ascending=[True, False, True])
    user_top_genres = (
        genre_counts.groupby("user_idx")["genres_list"].apply(lambda s: s.head(genre_top_k).tolist()).to_dict()
    )

    return genre_item_map, user_top_genres
