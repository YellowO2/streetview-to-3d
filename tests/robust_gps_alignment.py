"""Region-growing GPS alignment: a single least-squares fit across the
whole tree (tests/validate_gps_alignment.py) gets dragged off by
whichever chunk is badly wrong, corrupting the fit for the GOOD half
too. This instead seeds from one well-behaved chunk, greedily grows the
"confirmed good" set along real chunk adjacency, refitting after each
accepted chunk, and STOPS (rejects) at any chunk whose nodes don't match
the current fit -- that rejection point is the joint where the
reconstruction actually breaks, not just "some node somewhere."

Reads tests/validate_gps_alignment.py's own output (--out), which
already carries each node's raw da3_xz (pre-fit) and real_en, plus
chunk_id attribution.

Usage:
    python -m tests.robust_gps_alignment --in /tmp/gps_alignment.json --seed chunk0
"""
import argparse
import json
import os

import numpy as np

from street_builder.build_graph.global_dates import split_cover_into_chunks
from alignment.gps import fit_similarity_2d

NTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu")

ACCEPT_THRESHOLD_M = 30.0


def apply_fit(R, scale, t, pts):
    return scale * (pts @ R.T) + t


def chunk_adjacency():
    with open(os.path.join(NTU_DIR, "fetch_metadata.json")) as f:
        m = json.load(f)
    points = m["points"]
    adjacency = {int(k): v for k, v in m["adjacency"].items()}
    with open(os.path.join(NTU_DIR, "date_cover.json")) as f:
        cover = {int(k): v for k, v in json.load(f).items()}
    _, known_adjacent_chunk_pairs = split_cover_into_chunks(points, adjacency, cover, chunk_size=20)
    adj = {}
    for a, b in known_adjacent_chunk_pairs:
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    return adj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="/tmp/gps_alignment.json")
    parser.add_argument("--seed", default=None, help="chunk_id to start from; default: chunk with the most nodes")
    parser.add_argument("--threshold", type=float, default=ACCEPT_THRESHOLD_M)
    parser.add_argument("--out", default="/tmp/gps_alignment_robust.json")
    args = parser.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    by_chunk = {}
    for d in data:
        if d["chunk_id"] is None:
            continue
        by_chunk.setdefault(d["chunk_id"], []).append(d)

    adj = chunk_adjacency()
    print(f"{len(by_chunk)} chunk(s) with node data, adjacency graph has {len(adj)} chunk(s) with a real neighbor.")

    seed = args.seed or max(by_chunk, key=lambda c: len(by_chunk[c]))
    print(f"Seeding from {seed} ({len(by_chunk[seed])} node(s)).")

    confirmed = {seed}
    confirmed_nodes = list(by_chunk[seed])

    def current_fit():
        src = np.array([n["da3_xz"] for n in confirmed_nodes])
        dst = np.array([n["real_en"] for n in confirmed_nodes])
        return fit_similarity_2d(src, dst)

    R, scale, t = current_fit()

    # Iterative relaxation, not one-shot BFS: every pass, re-test the FULL
    # frontier (every chunk adjacent to ANY confirmed chunk, even ones a
    # previous pass rejected) against the latest fit. A chunk only counts
    # as a real "joint" once a full pass accepts nothing new -- rejecting
    # it once, from one neighbor, before the core (and the fit) had grown
    # enough isn't evidence of a break.
    pass_num = 0
    rejected = set()
    while True:
        pass_num += 1
        frontier = sorted({nb for c in confirmed for nb in adj.get(c, set())} - confirmed)
        if not frontier:
            rejected = set()
            break
        accepted_this_pass = []
        for cand in frontier:
            nodes = by_chunk.get(cand)
            if not nodes:
                continue
            src = np.array([n["da3_xz"] for n in nodes])
            dst = np.array([n["real_en"] for n in nodes])
            fitted = apply_fit(R, scale, t, src)
            residuals = np.linalg.norm(fitted - dst, axis=1)
            med = float(np.median(residuals))
            if med <= args.threshold:
                accepted_this_pass.append((cand, nodes, med))

        if not accepted_this_pass:
            rejected = set(frontier)
            print(f"pass {pass_num}: no frontier chunk fit -- stopping, {len(frontier)} chunk(s) at the boundary are real joints")
            break

        for cand, nodes, med in accepted_this_pass:
            confirmed.add(cand)
            confirmed_nodes.extend(nodes)
            print(f"  + {cand}: median residual {med:.1f}m <= {args.threshold:.0f}m -- accepted (pass {pass_num})")
        R, scale, t = current_fit()
        print(f"pass {pass_num}: accepted {len(accepted_this_pass)}, core now {len(confirmed)} chunk(s), refit.")

    all_chunks = set(by_chunk)
    unreached = all_chunks - confirmed - rejected
    print(f"\nDone. Confirmed (aligned) core: {len(confirmed)} chunk(s).")
    print(f"Rejected at the boundary: {sorted(rejected)}")
    print(f"Never reached (on the far side of a rejected joint, or disconnected): {len(unreached)} chunk(s): {sorted(unreached)}")

    # Final residuals for every node against the final confirmed-core fit,
    # for every chunk (confirmed, rejected, and unreached) so the whole
    # picture -- not just the core -- can be visualized.
    out = []
    for chunk_id, nodes in by_chunk.items():
        src = np.array([n["da3_xz"] for n in nodes])
        dst = np.array([n["real_en"] for n in nodes])
        fitted = apply_fit(R, scale, t, src)
        residuals = np.linalg.norm(fitted - dst, axis=1)
        status = "confirmed" if chunk_id in confirmed else ("rejected" if chunk_id in rejected else "unreached")
        for i, n in enumerate(nodes):
            out.append({**n, "fitted_en": fitted[i].tolist(), "residual_m": float(residuals[i]), "status": status})

    with open(args.out, "w") as f:
        json.dump({"nodes": out, "seed": seed, "confirmed": sorted(confirmed), "rejected": sorted(rejected),
                    "unreached": sorted(unreached), "threshold": args.threshold}, f)
    print(f"\nWrote {len(out)} node result(s) to {args.out}")


if __name__ == "__main__":
    main()
