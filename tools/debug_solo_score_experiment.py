"""One-time real-data experiment (kept for reference, NOT wired into the
live app): does a candidate's solo DA3 self-consistency score predict
pairwise DA3 success? And what does one real DA3 call actually cost,
wall-clock? See street_builder/reconstruction/walk_graph.py's
SECONDS_PER_DOT_ESTIMATE -- this is where its real-data calibration
comes from.

Not a pytest module -- GPU calls only run via a live Gradio queue on HF
Spaces (ZeroGPU), so re-running this for real requires temporarily
wiring run_debug_solo_score_experiment() into a button in
street_builder/tab.py again (same pattern debug_solo_score_experiment_gpu
below already follows), calling it, then removing the wiring again.

Results from the original run (2026-08-19, 3 real adjacent dot pairs on
a dense Apple LookAround date, 42 real GPU calls):
- DA3 model load: 8.93s
- Solo-score: 15 calls, avg 1.36s (min 1.27s, max 1.91s)
- Pairwise: 27 calls, avg 1.99s (min 1.89s, max 2.18s)
- Hypothesis CONFIRMED: pairwise success rate rose monotonically with
  the weaker candidate's solo score --
    min-score  6: 1/3  (33%)
    min-score  8: 2/3  (67%)
    min-score 11: 6/9  (67%)
    min-score 13+: all 12/12 (100%)
  Not deterministic (score 11 pairs split 6/9), but a real, useful
  signal -- consistent with score predicting LIKELIHOOD, not guaranteeing
  success.
"""
import os

from services.pipeline_runner import GPU_DISPATCH, get_da3_config
from services.streetview_fetch import run_async
from street_builder.build_graph.build_graph import build_corridor_graphs
from street_builder.main import DEFAULT_STEP_DEGREES, _download_all

# One real corridor (13 clicked nodes) known to have a dense Apple
# LookAround date (2026-06-16) with genuine multi-candidate dots, and
# several real adjacent dot pairs among them (found by inspecting
# build_corridor_graphs' own output for this corridor).
DEBUG_NODE_IDS = [
    "QauJOqDU81oP2gjvq7QU2w", "2bydvzQImenF9gtmHp5lXA", "6ji2HZUXlSPv3P10r1C5_Q",
    "zl35LNRK_aJe4cGQ9-hTRQ", "PKxYVot6hYv2bCE_U44HWw", "ZEZ0GsFosASfXe1ZjVlYzQ",
    "wqnEAA3J8vYGdz0hnN_raQ", "3kUzmNGgFgcUXgYoYFixhg", "BbUg0asjBBgWw4m2qtvV2w",
    "A-qJINmQuveuZjkfGWt22A", "cSZ1RgE0puHs8fldOqvUjA", "aUG1My3j2n0sZl9S_cOwTg", "XeZFUZK6aeLyoMOG7Z1bSA",
]
DEBUG_DATE = "2026-06-16"
# 6 adjacent multi-candidate pairs are available for this corridor+date:
# (0,1) (1,2) (2,5) (5,6) (6,7) (7,8) -- trimmed to 3 for the real run to
# keep GPU cost down (42 calls instead of 72).
DEBUG_ADJACENT_PAIRS = [(0, 1), (5, 6), (2, 5)]


@GPU_DISPATCH
def debug_solo_score_experiment_gpu(dot_pairs, step_degrees=DEFAULT_STEP_DEGREES):
    """The one @spaces.GPU call. dot_pairs: [([(key, path), ...],
    [(key, path), ...]), ...] -- each tuple is one real adjacent dot
    pair's own candidate lists (already downloaded). Every candidate in
    every pair gets solo-scored once; every (a, b) combination within
    each pair gets a real pairwise test.

    Returns {"load_time_s": float, "scores": {key: (kept_views, time_s)},
    "results": [(key_a, key_b, score_a, score_b, ok, time_s), ...]}."""
    import tempfile
    import time

    import torch
    from panoramic_da3 import DA3Model, extract_views_for_da3
    from services.da3_ops import test_edge as da3_test_edge

    cfg = get_da3_config()
    t0 = time.monotonic()
    da3 = DA3Model(cfg.da3_model)
    load_time_s = time.monotonic() - t0
    print(f"[debug] DA3Model load: {load_time_s:.2f}s")

    scores = {}
    results = []
    try:
        with tempfile.TemporaryDirectory() as views_base:
            all_candidates = {}
            for cands_a, cands_b in dot_pairs:
                for key, path in cands_a + cands_b:
                    all_candidates[key] = path
            n_solo = len(all_candidates)
            n_pairwise = sum(len(a) * len(b) for a, b in dot_pairs)
            print(f"[debug] plan: {n_solo} solo-score call(s), {n_pairwise} pairwise call(s), {n_solo + n_pairwise} total")

            for i, (key, path) in enumerate(all_candidates.items()):
                t0 = time.monotonic()
                da3_dir = os.path.join(views_base, f"score_{i}")
                os.makedirs(da3_dir, exist_ok=True)
                views = extract_views_for_da3(path, da3_dir, step_degrees=step_degrees, prefix=f"score_{i}_", pano_id=0)
                filtered_views, _ = da3.process_views(views, dist_thresh=0.2, angle_thresh=1)
                elapsed = time.monotonic() - t0
                scores[key] = (len(filtered_views), elapsed)
                print(f"[debug] solo-score {i + 1}/{n_solo}: {key}: {len(filtered_views)}/{len(views)} kept, {elapsed:.2f}s")

            test_id = 0
            for cands_a, cands_b in dot_pairs:
                for key_a, path_a in cands_a:
                    for key_b, path_b in cands_b:
                        t0 = time.monotonic()
                        result = da3_test_edge(path_a, path_b, cfg, views_base, da3,
                                                test_id=f"debug_{test_id}", step_degrees=step_degrees)
                        elapsed = time.monotonic() - t0
                        test_id += 1
                        ok = result is not None
                        score_a, _ = scores[key_a]
                        score_b, _ = scores[key_b]
                        print(f"[debug] pairwise {test_id}/{n_pairwise}: {key_a}(score={score_a}) x {key_b}(score={score_b}): "
                              f"{'OK' if ok else 'FAIL'}, {elapsed:.2f}s")
                        results.append((key_a, key_b, score_a, score_b, ok, elapsed))
    finally:
        del da3
        torch.cuda.empty_cache()

    return {"load_time_s": load_time_s, "scores": scores, "results": results}


def run_debug_solo_score_experiment() -> str:
    """CPU/network prep (fetch corridor, download candidates) + the one
    GPU call. Returns a markdown results table for display."""
    from services.streetview_fetch import fetch_pano_by_id

    async def _fetch_all():
        return {pid: await fetch_pano_by_id(pid) for pid in DEBUG_NODE_IDS}
    metas = run_async(_fetch_all())

    neighbor_ids = {pid: {n["id"] for n in metas[pid]["neighbors"]} for pid in DEBUG_NODE_IDS}
    edges = []
    for i, a in enumerate(DEBUG_NODE_IDS):
        for b in DEBUG_NODE_IDS[i + 1:]:
            if b in neighbor_ids[a] or a in neighbor_ids[b]:
                edges.append(((metas[a]["lat"], metas[a]["lon"]), (metas[b]["lat"], metas[b]["lon"])))

    start = (metas[DEBUG_NODE_IDS[0]]["lat"], metas[DEBUG_NODE_IDS[0]]["lon"])
    goals = [(metas[pid]["lat"], metas[pid]["lon"]) for pid in DEBUG_NODE_IDS[1:]]

    date_graphs, points, adjacency = build_corridor_graphs(edges, start[0], start[1], goals)
    date_graph = next((g for g in date_graphs if g["date"] == DEBUG_DATE), None)
    if date_graph is None:
        raise ValueError(f"date {DEBUG_DATE} not found among this corridor's top dates -- corridor data may have changed")
    dot_candidates = date_graph["dot_candidates"]

    dots_needed = {d for pair in DEBUG_ADJACENT_PAIRS for d in pair}
    all_nodes = [n for d in dots_needed for n in dot_candidates.get(d, [])]
    paths = run_async(_download_all(all_nodes))
    path_by_key = {n["key"]: p for n, p in zip(all_nodes, paths) if p}

    dot_pairs = []
    for a, b in DEBUG_ADJACENT_PAIRS:
        cands_a = [(n["key"], path_by_key[n["key"]]) for n in dot_candidates.get(a, []) if n["key"] in path_by_key]
        cands_b = [(n["key"], path_by_key[n["key"]]) for n in dot_candidates.get(b, []) if n["key"] in path_by_key]
        if cands_a and cands_b:
            dot_pairs.append((cands_a, cands_b))
    if not dot_pairs:
        raise ValueError("Nothing downloaded successfully -- can't run experiment")

    result = debug_solo_score_experiment_gpu(dot_pairs, step_degrees=DEFAULT_STEP_DEGREES)
    return format_debug_solo_score_results(result)


def format_debug_solo_score_results(result: dict) -> str:
    """Rolls up result["scores"]/result["results"] into a summary (timing
    aggregates + a direct successes-vs-failures score comparison) so the
    hypothesis doesn't have to be eyeballed out of a 50+ row table by
    hand."""
    scores = result["scores"]
    results = result["results"]
    solo_times = [t for _, t in scores.values()]
    pairwise_times = [r[5] for r in results]
    successes = [r for r in results if r[4]]
    failures = [r for r in results if not r[4]]

    def _avg(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def _min_score(r):
        return min(r[2], r[3])

    def _sum_score(r):
        return r[2] + r[3]

    summary = [
        "## Summary", "",
        f"- DA3 model load: {result['load_time_s']:.2f}s",
    ]
    if solo_times:
        summary.append(f"- Solo-score calls: {len(solo_times)}, avg {_avg(solo_times):.2f}s "
                        f"(min {min(solo_times):.2f}s, max {max(solo_times):.2f}s)")
    if pairwise_times:
        summary.append(f"- Pairwise calls: {len(pairwise_times)}, avg {_avg(pairwise_times):.2f}s "
                        f"(min {min(pairwise_times):.2f}s, max {max(pairwise_times):.2f}s)")
    summary.append(f"- Total experiment time: {result['load_time_s'] + sum(solo_times) + sum(pairwise_times):.2f}s")
    summary.append("")
    if results:
        summary.append(f"- Pairwise success rate: {len(successes)}/{len(results)} "
                        f"({100 * len(successes) / len(results):.0f}%)")
    if successes and failures:
        summary.append(f"- Avg weaker-side score (min of the pair) -- successes: {_avg([_min_score(r) for r in successes]):.1f}, "
                        f"failures: {_avg([_min_score(r) for r in failures]):.1f}")
        summary.append(f"- Avg combined score (sum of the pair) -- successes: {_avg([_sum_score(r) for r in successes]):.1f}, "
                        f"failures: {_avg([_sum_score(r) for r in failures]):.1f}")
    elif results:
        summary.append("- (every pair had the same outcome -- can't compare successes vs failures)")

    lines = summary + ["", "**Solo scores:**", "", "| pano | kept views | time (s) |", "|---|---|---|"]
    for key, (kept, elapsed) in scores.items():
        lines.append(f"| {key} | {kept} | {elapsed:.2f} |")
    lines += ["", "**Pairwise results:**", "", "| pano A | score A | pano B | score B | result | time (s) |", "|---|---|---|---|---|---|"]
    for key_a, key_b, score_a, score_b, ok, elapsed in results:
        lines.append(f"| {key_a} | {score_a} | {key_b} | {score_b} | {'OK' if ok else 'FAIL'} | {elapsed:.2f} |")
    return "\n".join(lines)
