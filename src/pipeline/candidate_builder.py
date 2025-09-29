from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import torch

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


def build_candidate_sources(
    train_df,
    num_users: int,
    user_consumed,
    covis_k: int = 150,
    itemcf_k: int = 150,
    popular_k: int = 100,
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

    return CandidateSources(
        covis_index=covis_index,
        item_cf_index=item_cf_index,
        popular_items=popular_items,
        covis_user_candidates=covis_user_candidates,
        itemcf_user_candidates=itemcf_user_candidates,
        popular_user_candidates=popular_user_candidates,
    )


def build_hybrid_candidates(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_consumed,
    sources: CandidateSources,
    candidate_k: int = 100,
    faiss_weight: float = 0.7,
    covis_weight: float = 0.4,
    itemcf_weight: float = 0.5,
    popular_weight: float = 0.3,
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

        user_candidates[u] = merge_candidate_lists(
            [
                (faiss_candidates, faiss_weight),
                (covis_list, covis_weight),
                (itemcf_list, itemcf_weight),
                (popular_list, popular_weight),
            ],
            candidate_k,
        )

    return user_candidates
