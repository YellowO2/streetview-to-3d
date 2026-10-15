"""Rank capture dates by the corridor each newly covers (no GPU)."""
from streetview_to_3d.panos.date_recency_key import date_recency_key


def rank_dates(buckets: dict[int, list[dict]]) -> list[str]:
    """Every date in any bucket, best first: most dots no earlier date covers, then most
    dots, then newest. New dots first, so an area's walked paths get a date even when
    its roads were driven many more times. Returns all dates; callers stop once enough pass.
    """
    covered_by_date: dict[str, set[int]] = {}
    for dot_index, bucket in buckets.items():
        for n in bucket:
            covered_by_date.setdefault(n["date"], set()).add(dot_index)

    ranked, covered = [], set()
    left = dict(covered_by_date)
    while left:
        date = max(left, key=lambda d: (len(left[d] - covered), len(left[d]), date_recency_key(d)))
        covered |= left.pop(date)
        ranked.append(date)
    return ranked
