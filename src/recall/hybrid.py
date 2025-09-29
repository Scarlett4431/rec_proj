from typing import Iterable, List, Sequence, Tuple


def merge_candidate_lists(
    sources: Sequence[Tuple[Iterable[int], float]],
    k: int,
) -> List[int]:
    """Blend multiple candidate sources using weighted scores instead of hard quotas."""

    if k <= 0 or not sources:
        return []

    processed: List[Tuple[List[int], float]] = []
    for items, weight in sources:
        if items is None:
            continue
        item_list = list(items)
        weight = float(weight)
        if not item_list or weight <= 0.0:
            continue
        processed.append((item_list, weight))

    if not processed:
        return []

    scores = {}
    best_rank = {}
    best_source = {}

    for source_idx, (items, weight) in enumerate(processed):
        for rank, item in enumerate(items):
            contribution = weight / (rank + 1.0)
            scores[item] = scores.get(item, 0.0) + contribution
            if item not in best_rank or rank < best_rank[item]:
                best_rank[item] = rank
                best_source[item] = source_idx

    ranked_items = sorted(
        scores.items(),
        key=lambda kv: (
            -kv[1],
            best_rank.get(kv[0], float("inf")),
            best_source.get(kv[0], float("inf")),
            kv[0],
        ),
    )

    return [item for item, _ in ranked_items[:k]]
