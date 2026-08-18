"""One-off diagnostic: real pairwise DA3 success rate between nearby
candidates of DIFFERENT dates, on the known 8-node test street. Informs
whether cross-date graph edges (not built yet, see walk_graph.py) are worth
pursuing. Delete this file + its button once the question's answered.

This street is on an elevated surface, where even same-date pairs have a
higher baseline failure rate than usual -- so a SAME_DATE_COUNT control
sample is included alongside the cross-date sample, both targeting the same
~TARGET_DIST_M distance, to tell "cross-date is worse" apart from "this
street just fails more often regardless."
"""
import datetime

from services.geo import haversine_m
from services.lookaround_fetch import DA3_ONLY_APPLE_ZOOM, download_lookaround
from services.pipeline_runner import test_pairs_gpu
from services.streetview_fetch import DA3_ONLY_ZOOM, run_async, download_pano_by_id
from street_builder.build_graph.build_graph import build_corridor_graph
from street_builder.reconstruction.best4 import BEST4_STEP_DEGREES

MAX_DIST_M = 10.0
TARGET_DIST_M = 5.0  # sample pairs closest to this distance, not just nearest overall
MAX_DATE_DIFF_YEARS = 2.0  # drop pairs further apart than this -- not a realistic drive gap
CROSS_DATE_COUNT = 20
SAME_DATE_COUNT = 6  # baseline control, same distance target

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


def _parse_date(s):
    parts = s.split("-")
    if len(parts) < 2 or not parts[0].isdigit():
        return None
    year, month = int(parts[0]), int(parts[1])
    day = int(parts[2]) if len(parts) > 2 else 15
    return datetime.date(year, month, day)


def _date_diff_years(a, b):
    da, db = _parse_date(a), _parse_date(b)
    if da is None or db is None:
        return None
    return abs((da - db).days) / 365.25


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
    """Returns human-readable result lines: a same-date baseline sample
    followed by the cross-date sample, both targeting ~TARGET_DIST_M."""
    nodes, edges, points = build_corridor_graph(WAYPOINTS)
    nodes = [n for n in nodes if edges.get(n["key"])]

    cross_pairs, same_pairs = [], []
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            d = haversine_m(a["lat"], a["lon"], b["lat"], b["lon"])
            if d > MAX_DIST_M:
                continue
            if a["date"] == b["date"]:
                same_pairs.append((d, a, b))
                continue
            diff_years = _date_diff_years(a["date"], b["date"])
            if diff_years is not None and diff_years > MAX_DATE_DIFF_YEARS:
                continue
            cross_pairs.append((d, a, b))

    cross_pairs.sort(key=lambda t: abs(t[0] - TARGET_DIST_M))
    same_pairs.sort(key=lambda t: abs(t[0] - TARGET_DIST_M))

    same_sample = _diverse_sample(same_pairs, SAME_DATE_COUNT)
    cross_sample = _diverse_sample(cross_pairs, CROSS_DATE_COUNT)
    if not same_sample and not cross_sample:
        return ["No pairs found within range."]

    groups = [("same-date baseline", same_sample), ("cross-date", cross_sample)]
    paths, labels, group_bounds = [], [], []
    for name, sample in groups:
        start = len(paths)
        for d, a, b in sample:
            paths.append((_download(a), _download(b)))
            labels.append(f"{d:.2f}m: {a['key']} ({a['date']}) <-> {b['key']} ({b['date']})")
        group_bounds.append((name, start, len(paths)))

    results = test_pairs_gpu(paths, step_degrees=BEST4_STEP_DEGREES) if paths else []

    lines = []
    for name, start, end in group_bounds:
        lines.append(f"-- {name} ({end - start} pairs) --")
        passed = 0
        for label, (ka, ta, kb, tb) in zip(labels[start:end], results[start:end]):
            healthy = (ka / ta >= 0.5) and (kb / tb >= 0.5)
            passed += healthy
            lines.append(f"{label} -> a={ka}/{ta} b={kb}/{tb} {'OK' if healthy else 'FAIL'}")
        total = end - start
        lines.append(f"{passed}/{total} passed (>=50% keep both sides)" if total else "no pairs found")
        lines.append("")
    return lines
