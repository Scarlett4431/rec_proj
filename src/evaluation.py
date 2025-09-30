import numpy as np
import torch
from src.faiss_index import FaissIndex
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


def recall_at_k(pred_items, true_item, k=10):
    return 1.0 if true_item in pred_items[:k] else 0.0

def ndcg_at_k(pred_items, true_item, k=10):
    if true_item in pred_items[:k]:
        idx = pred_items.index(true_item)
        return 1.0 / np.log2(idx + 2)
    return 0.0


def evaluate_candidates(candidate_map, test_pairs, k=100):
    """Evaluate recall metrics for arbitrary candidate pools."""
    hits = []
    ndcgs = []
    for u, true_item in test_pairs:
        candidates = candidate_map.get(u, [])[:k]
        if not candidates:
            hits.append(0.0)
            ndcgs.append(0.0)
            continue
        if true_item in candidates:
            idx = candidates.index(true_item)
            hits.append(1.0)
            ndcgs.append(1.0 / np.log2(idx + 2))
        else:
            hits.append(0.0)
            ndcgs.append(0.0)
    return {
        "recall@k": float(np.mean(hits)) if hits else 0.0,
        "ndcg@k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "hit_rate@k": float(np.mean(hits)) if hits else 0.0,
    }


def evaluate_filtered_faiss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    val_users: Iterable[int],
    val_pairs: List[tuple],
    user_consumed: Optional[Mapping[int, Iterable[int]]],
    num_items: int,
    k: int = 100,
):
    """FAISS evaluation that filters out training interactions per user."""

    val_users = list(val_users)
    if not val_users:
        return {"recall@k": 0.0, "ndcg@k": 0.0}

    consumed_map = user_consumed or {}
    index = FaissIndex(item_emb.detach().cpu())
    candidates: Dict[int, List[int]] = {}

    for u in val_users:
        consumed = consumed_map.get(u, set())
        if isinstance(consumed, list):
            consumed = set(consumed)
        search_k = min(num_items, k + len(consumed)) if consumed else k
        _, idxs = index.search(user_emb[u].detach().cpu(), k=search_k)
        filtered = [int(i) for i in idxs.tolist() if i >= 0 and i not in consumed]
        candidates[u] = filtered[:k]

    return evaluate_candidates(candidates, val_pairs, k=k)



def evaluate_ranker_with_candidates(
    ranker,
    user_emb,
    item_emb,
    user_candidates: Dict[int, List[int]],
    test_pairs,
    rank_k=10,
    device="cpu",
    user_feat_matrix=None,
    item_feat_matrix=None,
    user_histories: Dict[int, List[int]] | None = None,
    max_history: int = 50,
    batch_size: int = 4096,
):
    """Evaluate ranker using precomputed recall candidates."""
    if user_emb.device != device:
        user_emb = user_emb.to(device)
    if item_emb.device != device:
        item_emb = item_emb.to(device)
    if user_feat_matrix is not None and user_feat_matrix.device != device:
        user_feat_matrix = user_feat_matrix.to(device)
    if item_feat_matrix is not None and item_feat_matrix.device != device:
        item_feat_matrix = item_feat_matrix.to(device)

    results = {"recall@k": [], "ndcg@k": [], "gauc@k": []}

    requires_history = getattr(ranker, "requires_history", False)

    history_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    batch_size = max(batch_size, 1)

    if requires_history:
        zero_hist_emb = torch.zeros(
            (max_history, item_emb.shape[1]), dtype=item_emb.dtype, device=device
        )
        zero_hist_mask = torch.zeros((max_history,), dtype=torch.bool, device=device)

    def get_history_embeddings(user_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cached = history_cache.get(user_id)
        if cached is not None:
            return cached

        hist_items = user_histories.get(user_id, []) if user_histories is not None else []
        history_trim = hist_items[-max_history:]
        hist_tensor = torch.full((max_history,), -1, dtype=torch.long, device=device)
        if history_trim:
            hist_tensor[-len(history_trim):] = torch.as_tensor(
                history_trim, dtype=torch.long, device=device
            )
        hist_mask = hist_tensor >= 0
        hist_indices = hist_tensor.clone()
        hist_indices[~hist_mask] = 0
        hist_emb = item_emb[hist_indices]
        hist_emb = hist_emb * hist_mask.unsqueeze(-1)
        history_cache[user_id] = (hist_emb, hist_mask)
        return history_cache[user_id]

    pending_users: List[int] = []
    pending_true: List[int] = []
    pending_candidates: List[torch.Tensor] = []
    total_candidates = 0

    def process_pending():
        nonlocal total_candidates
        if not pending_users:
            return

        user_index_parts = []
        hist_emb_parts = []
        hist_mask_parts = []

        for user_id, cand_tensor in zip(pending_users, pending_candidates):
            count = cand_tensor.numel()
            user_index_parts.append(
                torch.full((count,), user_id, dtype=torch.long, device=device)
            )
            if requires_history:
                if user_histories is not None:
                    hist_emb_single, hist_mask_single = get_history_embeddings(user_id)
                else:
                    hist_emb_single, hist_mask_single = zero_hist_emb, zero_hist_mask
                hist_emb_parts.append(hist_emb_single.unsqueeze(0).expand(count, -1, -1))
                hist_mask_parts.append(hist_mask_single.unsqueeze(0).expand(count, -1))

        user_indices = torch.cat(user_index_parts, dim=0)
        item_indices = torch.cat(pending_candidates, dim=0)

        u_emb_batch = user_emb[user_indices]
        i_emb_batch = item_emb[item_indices]

        if (user_feat_matrix is not None) and (item_feat_matrix is not None):
            u_feat_batch = user_feat_matrix[user_indices]
            i_feat_batch = item_feat_matrix[item_indices]
        else:
            u_feat_batch = None
            i_feat_batch = None

        if requires_history:
            hist_emb_batch = torch.cat(hist_emb_parts, dim=0)
            hist_mask_batch = torch.cat(hist_mask_parts, dim=0)
        else:
            hist_emb_batch = None
            hist_mask_batch = None

        with torch.no_grad():
            if requires_history:
                scores = ranker(
                    u_emb_batch,
                    i_emb_batch,
                    u_feats=u_feat_batch,
                    i_feats=i_feat_batch,
                    hist_emb=hist_emb_batch,
                    hist_mask=hist_mask_batch,
                )
            else:
                scores = ranker(
                    u_emb_batch,
                    i_emb_batch,
                    u_feats=u_feat_batch,
                    i_feats=i_feat_batch,
                )

        scores = scores.view(-1)

        offset = 0
        for user_id, true_item, cand_tensor in zip(
            pending_users, pending_true, pending_candidates
        ):
            cand_count = cand_tensor.numel()
            user_scores = scores[offset : offset + cand_count]
            offset += cand_count

            if cand_count == 0:
                results["recall@k"].append(0.0)
                results["ndcg@k"].append(0.0)
                results["gauc@k"].append(0.0)
                continue

            top_k = min(rank_k, cand_count)
            top_indices = torch.topk(user_scores, top_k).indices
            ranked_items = cand_tensor[top_indices]

            hit_mask = ranked_items == true_item
            hit = 1.0 if hit_mask.any().item() else 0.0
            results["recall@k"].append(hit)

            if hit:
                rank_position = torch.nonzero(hit_mask, as_tuple=False)[0].item()
                ndcg = 1.0 / np.log2(rank_position + 2)
            else:
                ndcg = 0.0
            results["ndcg@k"].append(ndcg)

            pos_mask = cand_tensor == true_item
            if pos_mask.any().item() and cand_tensor.numel() > 1:
                pos_idx = torch.nonzero(pos_mask, as_tuple=False)[0].item()
                pos_score = user_scores[pos_idx]
                neg_scores = user_scores[~pos_mask]
                if neg_scores.numel() > 0:
                    better = torch.sum((pos_score > neg_scores).float()).item()
                    ties = torch.sum(
                        torch.isclose(pos_score.expand_as(neg_scores), neg_scores).float()
                    ).item()
                    auc = (better + 0.5 * ties) / neg_scores.numel()
                    results["gauc@k"].append(auc)
                else:
                    results["gauc@k"].append(0.0)
            else:
                results["gauc@k"].append(0.0)

        pending_users.clear()
        pending_true.clear()
        pending_candidates.clear()
        total_candidates = 0

    for u, true_item in test_pairs:
        candidate_items = user_candidates.get(u, [])
        if not candidate_items:
            results["recall@k"].append(0.0)
            results["ndcg@k"].append(0.0)
            results["gauc@k"].append(0.0)
            continue

        cand_tensor = torch.as_tensor(candidate_items, dtype=torch.long, device=device)
        pending_users.append(u)
        pending_true.append(true_item)
        pending_candidates.append(cand_tensor)
        total_candidates += cand_tensor.numel()

        if total_candidates >= batch_size:
            process_pending()

    process_pending()

    return {k: float(np.mean(v)) for k, v in results.items()}
