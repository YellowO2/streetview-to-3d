"""Downloads a connected ~N-chunk subset of an already-assembled tree-merge
group (e.g. g_L16_0) into a single plain, viewable local .ply -- for
dragging into a local viewer to sanity-check a manageable slice of a big
group without paying to re-run/re-merge anything (read-only: only
HfApi/hf_hub_download calls, no GPU billing).

Picks the subset by growing outward from one chunk along REAL
declared-adjacent pairs restricted to the group's own leaf chunk ids, so
the result is one geographically contiguous patch, not scattered chunks.

Usage:
    python -m tools.download_subset --group g_L16_0 --num-chunks 50 --out /tmp/subset.ply
"""
from paths import NTU_DIR
import argparse
import json
import os

import numpy as np

from street_builder.build_graph.global_dates import split_cover_into_chunks
from street_builder.reconstruction.join_segments import assemble_metadata_piece
from street_builder.tab import _leaf_ply_local_path, _load_group_meta_pieces

NTU_DIR = NTU_DIR


def _write_ply(path, pts, cols):
    n = len(pts)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    verts = np.zeros(n, dtype=np.dtype(fields))
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rgb = (np.clip(cols, 0, 1) * 255).astype("u1") if cols is not None else np.full((n, 3), 200, dtype="u1")
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


def grow_connected_subset(leaf_pairs, all_ids, n):
    """Same growth rule as tree_merge_test.py's grow_chain -- start from
    one real pair, repeatedly attach anything with exactly one end
    already in the chosen set, so every addition is a real adjacency to
    the growing patch."""
    relevant = [tuple(p) for p in leaf_pairs if p[0] in all_ids and p[1] in all_ids]
    if not relevant:
        return list(all_ids)[:n]
    a, b = relevant[0]
    chosen = [a, b]
    remaining = relevant[1:]
    while len(chosen) < n:
        grown = False
        for pair in remaining:
            x, y = pair
            if x in chosen and y not in chosen:
                chosen.append(y)
                remaining.remove(pair)
                grown = True
                break
            if y in chosen and x not in chosen:
                chosen.append(x)
                remaining.remove(pair)
                grown = True
                break
        if not grown:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--num-chunks", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(os.path.join(NTU_DIR, "fetch_metadata.json")) as f:
        m = json.load(f)
    points = m["points"]
    adjacency = {int(k): v for k, v in m["adjacency"].items()}
    with open(os.path.join(NTU_DIR, "date_cover.json")) as f:
        cover = {int(k): v for k, v in json.load(f).items()}
    _, known_adjacent_chunk_pairs = split_cover_into_chunks(points, adjacency, cover, chunk_size=args.chunk_size)

    print(f"Loading group {args.group}'s metadata...")
    meta_pieces = _load_group_meta_pieces(args.group)
    print(f"{len(meta_pieces)} piece(s) in group.")

    biggest = max(meta_pieces, key=lambda mp: len(mp[0]))
    all_ids = sorted({chunk_id for chunk_id, _, _, _ in biggest[0]})
    print(f"Biggest piece has {len(all_ids)} chunk(s), picking a connected subset of {args.num_chunks}...")

    subset = set(grow_connected_subset(known_adjacent_chunk_pairs, all_ids, args.num_chunks))
    print(f"Chosen {len(subset)} connected chunk(s): {sorted(subset)}")

    leaf_refs, path_edges, date, reached, frame_poses = biggest
    sub_leaf_refs = [ref for ref in leaf_refs if ref[0] in subset]
    sub_piece = (sub_leaf_refs, path_edges, date, reached, frame_poses)

    print("Downloading + assembling real point data for the subset (real chunk .ply data, no GPU calls)...")
    pts, cols, *_ = assemble_metadata_piece(sub_piece, _leaf_ply_local_path)
    print(f"Assembled {len(pts)} point(s). Writing to {args.out}...")
    _write_ply(args.out, pts, cols)
    print("Done.")


if __name__ == "__main__":
    main()
