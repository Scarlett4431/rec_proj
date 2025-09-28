from typing import Iterable, List, Sequence, Tuple


def merge_candidate_lists(
    sources: Sequence[Tuple[Iterable[int], float]],
    k: int,
) -> List[int]:
    """Blend multiple candidate sources according to provided ratios."""

    if k <= 0 or not sources:
        return []

    processed = [
        (list(items), max(0.0, ratio))
        for items, ratio in sources
        if items is not None
    ]
    if not processed:
        return []

    merged: List[int] = []
    seen = set()
    allocations = []
    for _, ratio in processed:
        allocations.append(int(ratio * k))
    total_alloc = sum(allocations)
    remainder = max(0, k - total_alloc)

    positions = [0] * len(processed)

    for idx, (items, _ratio) in enumerate(processed):
        target = min(allocations[idx], k - len(merged))
        while positions[idx] < len(items) and len(merged) < k and target > 0:
            candidate = items[positions[idx]]
            positions[idx] += 1
            if candidate in seen:
                continue
            merged.append(candidate)
            seen.add(candidate)
            target -= 1

    while remainder > 0 and len(merged) < k:
        progressed = False
        for idx, (items, _ratio) in enumerate(processed):
            if positions[idx] >= len(items):
                continue
            candidate = items[positions[idx]]
            positions[idx] += 1
            if candidate in seen:
                continue
            merged.append(candidate)
            seen.add(candidate)
            remainder -= 1
            progressed = True
            if remainder == 0 or len(merged) >= k:
                break
        if not progressed:
            break

    for idx, (items, _ratio) in enumerate(processed):
        while len(merged) < k and positions[idx] < len(items):
            candidate = items[positions[idx]]
            positions[idx] += 1
            if candidate in seen:
                continue
            merged.append(candidate)
            seen.add(candidate)

    return merged
