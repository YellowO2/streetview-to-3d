"""One-off diagnostic: real pairwise DA3 success rate between nearby
candidates of DIFFERENT dates, on the known 8-node test street. Informs
whether cross-date graph edges (not built yet, see walk_graph.py) are worth
pursuing. Delete this file + its button once the question's answered.
"""
from services.geo import haversine_m
from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, download_lookaround
from services.pipeline_runner import test_pairs_gpu
from services.streetview_fetch import DA3_ONLY_ZOOM, run_async, download_pano_by_id
from street_builder.build_graph.build_graph import build_corridor_graph
from street_builder.reconstruction.best4 import BEST4_STEP_DEGREES

MAX_DIST_M = 10.0
SAME_SPOT_COUNT = 4  # same physical stop, different capture years (0m apart)
SEPARATED_COUNT = 4  # genuinely different position, different date, still close

WAYPOINTS = [
    (1.3474607, 103.6837688),
    (1.347371899061369, 103.6837874097177),
    (1.347287860062261, 103.6838307916074),
    (1.347190300951547, 103.6838235465658),
    (1.347098788562233, 103.6838350326076),
    (1.347008631600778, 103.6838347675451),
    (1.346918186322612, 103.6838223979616),
    (1.346828812746741, 103.6837978355066),
]


def _download(node):
    if node["source"] == "apple":
        return download_lookaround(node["_pano"], zoom=DA3_ONLY_APPLE_ZOOM)
    return run_async(download_pano_by_id(node["id"], zoom=DA3_ONLY_ZOOM))


def _diverse_sample(pairs, n):
    """First n pairs, skipping repeats of the same physical location."""
    seen = set()
    picked = []
    for p in pairs:
        key = (round(p[1]["lat"], 5), round(p[1]["lon"], 5))
        if key in seen:
            continue
        seen.add(key)
        picked.append(p)
        if len(picked) >= n:
            break
    return picked


def test_cross_date_success_rate() -> list[str]:
    """Returns human-readable result lines, one per tested pair."""
    nodes, edges, points = build_corridor_graph(WAYPOINTS)
    nodes = [n for n in nodes if edges.get(n["key"])]

    cross_pairs = []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if a["date"] == b["date"]:
                continue
            d = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
            if d <= MAX_DIST_M:
                cross_pairs.append((d, a, b))
    cross_pairs.sort(key=lambda t: t[0])

    same_spot = [p for p in cross_pairs if p[0] < 0.5]
    separated = sorted((p for p in cross_pairs if p[0] >= 0.5), key=lambda t: -t[0])

    sample = _diverse_sample(same_spot, SAME_SPOT_COUNT) + _diverse_sample(separated, SEPARATED_COUNT)
    if not sample:
        return ["No cross-date pairs found within range."]

    paths, labels = [], []
    for d, a, b in sample:
        paths.append((_download(a), _download(b)))
        labels.append(f"{d:.2f}m: {a['key']} ({a['date']}) <-> {b['key']} ({b['date']})")

    results = test_pairs_gpu(paths, step_degrees=BEST4_STEP_DEGREES)

    lines = []
    passed = 0
    for label, (ka, ta, kb, tb) in zip(labels, results):
        healthy = (ka / ta >= 0.5) and (kb / tb >= 0.5)
        passed += healthy
        lines.append(f"{label} -> a={ka}/{ta} b={kb}/{tb} {'OK' if healthy else 'FAIL'}")
    lines.append(f"\n{passed}/{len(results)} pairs passed (>=50% keep both sides)")
    return lines
