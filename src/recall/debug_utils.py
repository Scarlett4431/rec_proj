from typing import Callable, Mapping, Sequence


def inspect_recall_result(
    user_id: int,
    target_items: int | Sequence[int] | None,
    candidate_items: Sequence[int] | None,
    id_to_name: Mapping[int, str],
    user_history: Sequence[int] | None = None,
    history_k: int | None = 10,
    top_k: int | None = 10,
    output_fn: Callable[[str], None] | None = None,
) -> str:
    """Format recall outputs with human-readable item names for debugging."""

    target_list: list[int]
    if target_items is None:
        target_list = []
    elif isinstance(target_items, int):
        target_list = [int(target_items)]
    else:
        target_list = [int(t) for t in target_items]

    candidate_list: list[int] = []
    if candidate_items:
        candidate_list = [int(c) for c in candidate_items]

    if top_k is not None and top_k >= 0:
        candidate_view = candidate_list[:top_k]
    else:
        candidate_view = candidate_list

    def resolve_name(idx: int) -> str:
        name = id_to_name.get(idx)
        return name if name is not None and name != "" else "<unknown>"

    lines = [f"[RecallDebug] user={user_id}"]

    history_list: list[int] = []
    if user_history:
        history_list = [int(h) for h in user_history]
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

    if target_list:
        for t in target_list:
            lines.append(f"  target {t}: {resolve_name(t)}")
    else:
        lines.append("  target: <not provided>")

    if candidate_view:
        lines.append("  candidates:")
        for rank, item_id in enumerate(candidate_view, start=1):
            lines.append(f"    {rank:02d}. {item_id}: {resolve_name(item_id)}")
        if len(candidate_view) < len(candidate_list):
            lines.append(
                f"    ... truncated {len(candidate_list) - len(candidate_view)} of {len(candidate_list)} total"
            )
    else:
        lines.append("  candidates: <empty>")

    message = "\n".join(lines)

    callback = output_fn or print
    callback(message)
    return message
