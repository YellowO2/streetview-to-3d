"""Measure how many metres one DA3 unit is -- the constant in config.py.

DA3 is internally consistent: the same unit means the same length in every
reconstruction, so one number converts its output to metres. Fitting a whole
chunk to its GPS does NOT measure it, because a chunk that drifted during
reconstruction reports a scale inflated by its own bend. So each chunk is
first cut wherever its own nodes stop matching their GPS, and only the
surviving pieces are fitted.

    python -m tools.measure_da3_scale --group g_L16_0

Pieces under MIN_NODES are ignored: a 2-node fit is exactly determined and
so reports zero residual whatever the data, and a 3-node one barely
constrains scale at all -- including them triples the scatter without
moving the answer.

Tightening --threshold trims the sample rather than changing the result:
measured on NTU the answer held at 1.334-1.346 from 12 m down to 0.75 m.
The sweep is worth reading, which is why --sweep prints one row per
threshold. Where the mean meets the median, nothing is skewing what is
left, and that is the value to take.
"""
import argparse
import json

import numpy as np

from postprocess.gps_fit.fit import fit_similarity_2d
from postprocess.gps_fit.discover_pieces import resolve
from postprocess.gps_fit.split_by_fit import _adjacency
from services.geo import latlon_to_local_m

MIN_NODES = 4
THRESHOLDS = (12, 8, 5, 3, 2, 1.5, 1.0, 0.75)


def nodes_from_group(group_id):
    """{key: {da3_xz, real_en, chunk_id}} for every node in a merged group."""
    from ui.tab import _load_group_meta_pieces
    from tools.validate_gps_alignment import build_key_to_chunk_map

    key_to_chunk = build_key_to_chunk_map()
    out = {}
    for _, _, _, _, frame_poses in _load_group_meta_pieces(group_id):
        keys = list(frame_poses)
        if len(keys) < MIN_NODES:
            continue
        lat0, lon0 = frame_poses[keys[0]][3], frame_poses[keys[0]][4]
        for k in keys:
            pos, _, _, lat, lon, *_ = frame_poses[k]
            out[k] = {"da3_xz": [pos[0], pos[2]],
                      "real_en": list(latlon_to_local_m(lat, lon, lat0, lon0)),
                      "chunk_id": key_to_chunk.get(k)}
    return out


def nodes_from_file(path):
    """The same, from a validate_gps_alignment dump."""
    with open(path) as f:
        return {d["key"]: d for d in json.load(f)}


def scales(nodes, threshold):
    """Every surviving piece's own DA3-to-metres estimate."""
    by_chunk = {}
    for k, d in nodes.items():
        by_chunk.setdefault(d.get("chunk_id"), {})[k] = d

    out = []
    for group in by_chunk.values():
        for piece in resolve(group, _adjacency(group), threshold=threshold):
            if len(piece) < MIN_NODES:
                continue
            src = np.array([nodes[k]["da3_xz"] for k in piece])
            dst = np.array([nodes[k]["real_en"] for k in piece])
            R, s, t = fit_similarity_2d(src, dst)
            res = np.linalg.norm(s * (src @ R.T) + t - dst, axis=1)
            out.append((len(piece), float(np.median(res)), float(s)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--group", help="a merged group id on the Hub, e.g. g_L16_0")
    src.add_argument("--in", dest="in_path", help="a validate_gps_alignment dump")
    ap.add_argument("--threshold", type=float, default=0.75,
                    help="metres a node may sit from its GPS before it is cut off")
    ap.add_argument("--sweep", action="store_true", help="one row per threshold")
    args = ap.parse_args()

    nodes = nodes_from_file(args.in_path) if args.in_path else nodes_from_group(args.group)
    print(f"{len(nodes)} node(s)\n")
    print(f"{'cut':>6} {'pieces':>7} {'nodes':>6} {'median':>8} {'mean':>8} {'sd':>7}")
    for th in (THRESHOLDS if args.sweep else (args.threshold,)):
        rows = scales(nodes, th)
        if len(rows) < 3:
            print(f"{th:>6} {len(rows):>7}   too few pieces to measure")
            continue
        a = np.array([s for _, _, s in rows])
        print(f"{th:>6} {len(rows):>7} {sum(n for n, _, _ in rows):>6} "
              f"{np.median(a):>8.4f} {a.mean():>8.4f} {a.std():>7.4f}")


if __name__ == "__main__":
    main()
