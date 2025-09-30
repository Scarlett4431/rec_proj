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

    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)
    if user_feat_matrix is not None:
        user_feat_matrix = user_feat_matrix.to(device)
    if item_feat_matrix is not None:
        item_feat_matrix = item_feat_matrix.to(device)

    results = {"recall@k": [], "ndcg@k": [], "gauc@k": []}

    for u, true_item in test_pairs:
        candidate_items = user_candidates.get(u, [])
        if not candidate_items:
            results["recall@k"].append(0.0)
            results["ndcg@k"].append(0.0)
            continue

        cand_vecs = item_emb[candidate_items]
        u_vec = user_emb[u].unsqueeze(0)
        u_expand = u_vec.repeat(cand_vecs.size(0), 1)

        u_side = i_side = None
        if (user_feat_matrix is not None) and (item_feat_matrix is not None):
            u_side = user_feat_matrix[u].unsqueeze(0).repeat(cand_vecs.size(0), 1)
            i_side = item_feat_matrix[candidate_items]

        hist_emb = hist_mask = None
        if getattr(ranker, "requires_history", False) and user_histories is not None:
            history_items = user_histories.get(u, [])
            history_trim = history_items[-max_history:]
            hist_tensor = torch.full((max_history,), -1, dtype=torch.long, device=device)
            if history_trim:
                hist_tensor[-len(history_trim):] = torch.tensor(history_trim, dtype=torch.long, device=device)
            hist_mask = hist_tensor >= 0
            hist_indices = hist_tensor.clone()
            hist_indices[~hist_mask] = 0
            hist_emb_base = item_emb[hist_indices.long()]
            hist_emb_base = hist_emb_base * hist_mask.unsqueeze(-1)
            hist_emb = hist_emb_base.unsqueeze(0).expand(cand_vecs.size(0), -1, -1)
            hist_mask = hist_mask.unsqueeze(0).expand(cand_vecs.size(0), -1)

        with torch.no_grad():
            if getattr(ranker, "requires_history", False):
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
            scores = scores_tensor.view(-1).detach().cpu().numpy()

        ranked_idx = np.argsort(-scores)[:rank_k]
        ranked_items = [candidate_items[i] for i in ranked_idx]

        results["recall@k"].append(recall_at_k(ranked_items, true_item, rank_k))
        results["ndcg@k"].append(ndcg_at_k(ranked_items, true_item, rank_k))

        if true_item in candidate_items and len(candidate_items) > 1:
            pos_idx = candidate_items.index(true_item)
            pos_score = scores[pos_idx]
            neg_scores = np.delete(scores, pos_idx)
            if neg_scores.size > 0:
                better = np.sum(pos_score > neg_scores)
                ties = np.sum(np.isclose(pos_score, neg_scores))
                auc = (better + 0.5 * ties) / neg_scores.size
                results["gauc@k"].append(float(auc))

    return {k: float(np.mean(v)) for k, v in results.items()}
