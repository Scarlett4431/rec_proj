import numpy as np
import torch
from src.faiss_index import FaissIndex
from typing import Dict, Iterable, List, Mapping, Optional


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

    for u, true_item in test_pairs:
        candidate_items = user_candidates.get(u, [])
        if not candidate_items:
            results["recall@k"].append(0.0)
            results["ndcg@k"].append(0.0)
            results["gauc@k"].append(0.0)
            continue

        candidate_tensor = torch.as_tensor(candidate_items, dtype=torch.long, device=device)
        cand_vecs = item_emb[candidate_tensor]
        u_vec = user_emb[u].unsqueeze(0)
        u_expand = u_vec.repeat(cand_vecs.size(0), 1)

        u_side = i_side = None
        if (user_feat_matrix is not None) and (item_feat_matrix is not None):
            u_side = user_feat_matrix[u].unsqueeze(0).repeat(cand_vecs.size(0), 1)
            i_side = item_feat_matrix[candidate_tensor]

        hist_emb = hist_mask = None
        if requires_history and user_histories is not None:
            history_items = user_histories.get(u, [])
            history_trim = history_items[-max_history:]
            hist_tensor = torch.full((max_history,), -1, dtype=torch.long, device=device)
            if history_trim:
                hist_tensor[-len(history_trim):] = torch.as_tensor(history_trim, dtype=torch.long, device=device)
            hist_mask_single = hist_tensor >= 0
            hist_indices = hist_tensor.clone()
            hist_indices[~hist_mask_single] = 0
            hist_emb_base = item_emb[hist_indices.long()]
            hist_emb_base = hist_emb_base * hist_mask_single.unsqueeze(-1)
            hist_emb = hist_emb_base.unsqueeze(0).expand(cand_vecs.size(0), -1, -1)
            hist_mask = hist_mask_single.unsqueeze(0).expand(cand_vecs.size(0), -1)

        with torch.no_grad():
            if requires_history:
                scores_tensor = ranker(
                    u_expand,
                    cand_vecs,
                    u_feats=u_side,
                    i_feats=i_side,
                    hist_emb=hist_emb,
                    hist_mask=hist_mask,
                )
            else:
                scores_tensor = ranker(
                    u_expand,
                    cand_vecs,
                    u_feats=u_side,
                    i_feats=i_side,
                )
            scores_tensor = scores_tensor.view(-1)

        top_k = min(rank_k, scores_tensor.size(0))
        top_indices = torch.topk(scores_tensor, top_k).indices
        ranked_items_tensor = candidate_tensor[top_indices]
        ranked_items = ranked_items_tensor.tolist()

        hit = 1.0 if true_item in ranked_items else 0.0
        results["recall@k"].append(hit)
        if hit:
            rank_position = ranked_items.index(true_item)
            ndcg = 1.0 / np.log2(rank_position + 2)
        else:
            ndcg = 0.0
        results["ndcg@k"].append(ndcg)

        pos_mask = candidate_tensor == true_item
        if pos_mask.any() and candidate_tensor.numel() > 1:
            pos_idx = torch.nonzero(pos_mask, as_tuple=False)[0].item()
            pos_score = scores_tensor[pos_idx]
            neg_scores = scores_tensor[~pos_mask]
            if neg_scores.numel() > 0:
                better = torch.sum((pos_score > neg_scores).float()).item()
                ties = torch.sum(torch.isclose(pos_score.expand_as(neg_scores), neg_scores).float()).item()
                auc = (better + 0.5 * ties) / neg_scores.numel()
                results["gauc@k"].append(auc)
            else:
                results["gauc@k"].append(0.0)
        else:
            results["gauc@k"].append(0.0)

    return {k: float(np.mean(v)) for k, v in results.items()}
