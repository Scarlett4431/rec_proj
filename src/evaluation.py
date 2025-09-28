import numpy as np
import torch
from src.faiss_index import FaissIndex

def recall_at_k(pred_items, true_item, k=10):
    return 1.0 if true_item in pred_items[:k] else 0.0

def ndcg_at_k(pred_items, true_item, k=10):
    if true_item in pred_items[:k]:
        idx = pred_items.index(true_item)
        return 1.0 / np.log2(idx + 2)
    return 0.0

def evaluate_with_faiss(user_emb, item_emb, test_pairs, k=10):
    index = FaissIndex(item_emb)
    results = {"recall@k": [], "ndcg@k": []}
    for u, true_item in test_pairs:
        _, idxs = index.search(user_emb[u], k=k)
        ranked_items = idxs.tolist()
        results["recall@k"].append(recall_at_k(ranked_items, true_item, k))
        results["ndcg@k"].append(ndcg_at_k(ranked_items, true_item, k))
    return {k: float(np.mean(v)) for k, v in results.items()}

def evaluate_ranker_with_faiss(
    ranker,
    user_emb,                # torch.Tensor [U+1, d]
    item_emb,                # torch.Tensor [I+1, d]
    test_pairs,              # list[(u, true_i)]
    faiss_k=200,
    rank_k=10,
    device="cpu",
    user_feat_matrix=None,   # torch.Tensor [U+1, Fu] or None
    item_feat_matrix=None    # torch.Tensor [I+1, Fi] or None
):
    """
    Recall + Rank evaluation with side features:
      1) FAISS to get faiss_k candidates per user
      2) Build extra features for (user, candidate_items)
      3) Rank with RankerMLP(user_emb, item_emb, extra_feats)
      4) Compute metrics at rank_k
    """
    # Move tensors once
    user_emb = user_emb.to(device)
    item_emb = item_emb.to(device)
    if user_feat_matrix is not None:
        user_feat_matrix = user_feat_matrix.to(device)
    if item_feat_matrix is not None:
        item_feat_matrix = item_feat_matrix.to(device)

    index = FaissIndex(item_emb.detach().cpu())  # faiss works on CPU float32
    results = {"recall@k": [], "ndcg@k": []}

    for u, true_item in test_pairs:
        # 1) Recall candidates
        _, idxs = index.search(user_emb[u].detach().cpu(), k=faiss_k)
        candidate_items = [i for i in idxs.tolist() if i >= 0]
        if not candidate_items:
            results["recall@k"].append(0.0)
            results["ndcg@k"].append(0.0)
            continue

        # 2) Build inputs for ranker
        u_vec = user_emb[u].unsqueeze(0)                         # [1, d]
        cand_vecs = item_emb[candidate_items]                    # [C, d]
        u_expand = u_vec.repeat(cand_vecs.size(0), 1)            # [C, d]

        # Side features (optional)
        u_side = i_side = None
        if (user_feat_matrix is not None) and (item_feat_matrix is not None):
            u_side = user_feat_matrix[u].unsqueeze(0).repeat(cand_vecs.size(0), 1)  # [C, Fu]
            i_side = item_feat_matrix[candidate_items]                               # [C, Fi]

        # 3) Rank
        with torch.no_grad():
            scores = ranker(u_expand, cand_vecs,
                           u_feats=u_side,
                           i_feats=i_side)
            # ensure 1D
            scores = scores.view(-1).detach().cpu().numpy()

        ranked_idx = np.argsort(-scores)[:rank_k]
        ranked_items = [candidate_items[i] for i in ranked_idx]

        # 4) Metrics
        results["recall@k"].append(recall_at_k(ranked_items, true_item, rank_k))
        results["ndcg@k"].append(ndcg_at_k(ranked_items, true_item, rank_k))

    return {k: float(np.mean(v)) for k, v in results.items()}
