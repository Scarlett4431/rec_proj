from typing import Callable, Iterable, Mapping, Sequence

import torch


def _ensure_device(tensor: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
    if tensor is None:
        return None
    if tensor.device == device:
        return tensor
    return tensor.to(device)


def score_ranker_candidates(
    ranker,
    user_id: int,
    candidate_items: Sequence[int],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_feat_matrix: torch.Tensor | None = None,
    item_feat_matrix: torch.Tensor | None = None,
    user_histories: Mapping[int, Sequence[int]] | None = None,
    max_history: int = 50,
    device: torch.device | None = None,
) -> list[tuple[int, float]]:
    """Return candidate scores for a single user using the trained ranker."""

    if not candidate_items:
        return []

    if device is None:
        try:
            device = next(ranker.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    candidate_list = [int(c) for c in candidate_items]
    cand_tensor = torch.tensor(candidate_list, dtype=torch.long, device=device)
    user_indices = torch.full((len(candidate_list),), int(user_id), dtype=torch.long, device=device)

    user_emb_device = _ensure_device(user_emb, device)
    item_emb_device = _ensure_device(item_emb, device)

    u_emb_batch = user_emb_device[user_indices]
    i_emb_batch = item_emb_device[cand_tensor]

    u_feat_batch = None
    if user_feat_matrix is not None and user_feat_matrix.numel() > 0:
        user_feat_vec = user_feat_matrix[int(user_id)]
        u_feat_batch = user_feat_vec.to(device).unsqueeze(0).expand(len(candidate_list), -1)

    i_feat_batch = None
    if item_feat_matrix is not None and item_feat_matrix.numel() > 0:
        item_feats = item_feat_matrix[cand_tensor.cpu()]
        i_feat_batch = item_feats.to(device)

    requires_history = getattr(ranker, "requires_history", False)
    if requires_history:
        history_seq: Sequence[int] = ()
        if user_histories is not None:
            history_seq = user_histories.get(int(user_id), ())
        history_trim = list(history_seq)[-max_history:]
        hist_tensor = torch.full((max_history,), -1, dtype=torch.long, device=device)
        if history_trim:
            hist_tensor[-len(history_trim) :] = torch.tensor(history_trim, dtype=torch.long, device=device)
        hist_mask = hist_tensor >= 0
        hist_indices = hist_tensor.clone()
        hist_indices[~hist_mask] = 0
        hist_emb_single = item_emb_device[hist_indices]
        hist_emb_single = hist_emb_single * hist_mask.unsqueeze(-1)
        hist_emb_batch = hist_emb_single.unsqueeze(0).expand(len(candidate_list), -1, -1)
        hist_mask_batch = hist_mask.unsqueeze(0).expand(len(candidate_list), -1)
    else:
        hist_emb_batch = None
        hist_mask_batch = None

    was_training = ranker.training
    ranker.eval()
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
    if was_training:
        ranker.train()

    score_list = scores.view(-1).tolist()
    ranked = sorted(zip(candidate_list, score_list), key=lambda kv: kv[1], reverse=True)
    return ranked


def inspect_rank_result(
    ranker,
    user_id: int,
    candidate_items: Sequence[int],
    id_to_name: Mapping[int, str],
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    user_feat_matrix: torch.Tensor | None = None,
    item_feat_matrix: torch.Tensor | None = None,
    user_histories: Mapping[int, Iterable[int]] | None = None,
    gold_items: Sequence[int] | int | None = None,
    max_history: int = 50,
    history_k: int | None = 10,
    top_k: int | None = 10,
    device: torch.device | None = None,
    output_fn: Callable[[str], None] | None = None,
) -> str:
    """Pretty-print ranker scores for a single user."""

    scored = score_ranker_candidates(
        ranker,
        user_id,
        candidate_items,
        user_emb,
        item_emb,
        user_feat_matrix=user_feat_matrix,
        item_feat_matrix=item_feat_matrix,
        user_histories=user_histories,
        max_history=max_history,
        device=device,
    )

    gold_list: list[int]
    if gold_items is None:
        gold_list = []
    elif isinstance(gold_items, int):
        gold_list = [int(gold_items)]
    else:
        gold_list = [int(g) for g in gold_items]

    gold_set = set(gold_list)

    def resolve_name(idx: int) -> str:
        name = id_to_name.get(idx)
        return name if name else "<unknown>"

    lines = [f"[RankDebug] user={user_id}"]

    history_seq: Sequence[int] = ()
    if user_histories is not None:
        history_seq = user_histories.get(int(user_id), ())
    history_view: Sequence[int]
    if history_seq:
        history_list = list(history_seq)
        if history_k is not None and history_k >= 0:
            history_view = history_list[-history_k:]
        else:
            history_view = history_list
        if history_view:
            lines.append("  recent history (oldest→newest):")
            for item_id in history_view:
                lines.append(f"    * {item_id}: {resolve_name(item_id)}")
        else:
            lines.append("  recent history: <empty>")
    else:
        lines.append("  recent history: <none>")

    if gold_list:
        lines.append("  gold items:")
        for item_id in gold_list:
            lines.append(f"    - {item_id}: {resolve_name(item_id)}")
    else:
        lines.append("  gold items: <none>")

    if top_k is not None and top_k >= 0:
        scored_view = scored[:top_k]
    else:
        scored_view = scored

    if scored_view:
        lines.append("  ranked candidates:")
        for rank, (item_id, score) in enumerate(scored_view, start=1):
            marker = " *" if item_id in gold_set else ""
            lines.append(f"    {rank:02d}. {item_id}: {resolve_name(item_id)} (score={score:.4f}){marker}")
        if len(scored_view) < len(scored):
            lines.append(
                f"    ... truncated {len(scored) - len(scored_view)} of {len(scored)} total"
            )
    else:
        lines.append("  ranked candidates: <empty>")

    message = "\n".join(lines)
    (output_fn or print)(message)
    return message
