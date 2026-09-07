"""Exports one island (from tools/piece_gps_test.py's output) as a real,
GPS-corrected .ply -- applies the island's own best-fit (rotation, scale,
translation) on top of the group's already-bridged frame, so the result
sits in its true GPS-implied position/orientation/scale rather than
wherever the original (possibly drifted) bridge chain left it.

Only works cleanly for islands where every touched chunk is 100% included
(no chunk split across islands) -- check with piece_gps_test's own output
first; a partially-included chunk has no way to extract just its
contributing nodes' points from the chunk's already-merged raw .ply.

Usage:
    python -m tools.export_island_ply --island 12 --group g_L16_0 --out ~/Downloads/island12.ply
"""
import argparse
import json
import os

import numpy as np

from street_builder.reconstruction.join_segments import assemble_metadata_piece
from street_builder.tab import _leaf_ply_local_path, _load_group_meta_pieces
from postprocess.gps_fit.fit import fit_similarity_2d


def write_ply(path, pts, cols):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--island", type=int, required=True)
    parser.add_argument("--group", default="g_L16_0")
    parser.add_argument("--pieces-json", default="/tmp/gps_pieces.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.pieces_json) as f:
        pieces_data = json.load(f)
    island_nodes = [d for d in pieces_data if d["island"] == args.island]
    if not island_nodes:
        raise SystemExit(f"No node(s) found for island {args.island} in {args.pieces_json}")
    island_chunks = sorted({d["chunk_id"] for d in island_nodes})
    print(f"Island {args.island}: {len(island_nodes)} node(s) across {len(island_chunks)} chunk(s): {island_chunks}")

    print(f"Loading group {args.group}'s metadata...")
    meta_pieces = _load_group_meta_pieces(args.group)
    biggest = max(meta_pieces, key=lambda mp: len(mp[0]))
    leaf_refs, path_edges, date, reached, frame_poses = biggest

    island_leaf_refs = [ref for ref in leaf_refs if ref[0] in island_chunks]
    found_chunks = {ref[0] for ref in island_leaf_refs}
    missing = set(island_chunks) - found_chunks
    if missing:
        raise SystemExit(f"Chunk(s) {missing} from island {args.island} not found in group {args.group}'s leaf_refs -- wrong group?")

    island_keys = {d["key"] for d in island_nodes}
    fit_nodes = [d for d in island_nodes if d["key"] in frame_poses]
    src = np.array([d["da3_xz"] for d in fit_nodes])
    dst = np.array([d["real_en"] for d in fit_nodes])
    R2, scale2, t2 = fit_similarity_2d(src, dst)
    print(f"Island fit: scale={scale2:.3f}")

    sub_piece = (island_leaf_refs, path_edges, date, reached, frame_poses)
    print(f"Downloading + assembling real point data for {len(island_leaf_refs)} chunk(s) (no GPU calls)...")
    pts, cols, *_ = assemble_metadata_piece(sub_piece, _leaf_ply_local_path)

    xz = pts[:, [0, 2]] @ R2.T * scale2 + t2
    corrected = np.column_stack([xz[:, 0], pts[:, 1], xz[:, 1]])

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"Assembled {len(corrected)} point(s). Writing to {out_path}...")
    write_ply(out_path, corrected, cols)
    print("Done.")


if __name__ == "__main__":
    main()
