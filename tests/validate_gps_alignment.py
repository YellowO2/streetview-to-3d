"""Sanity-checks a merged group's DA3 reconstruction against real GPS:
fits ONE similarity transform (rotation + scale + translation -- unlike
tests/gps.py's old per-segment _fit_rigid_2d, this is fit ONCE across the
WHOLE bridged tree) mapping every node's local (x, z) position onto its
real lat/lon (converted to local meters), then reports the residual
(post-fit distance error) per node. A node whose DA3-reconstructed
position doesn't match its real GPS position even after the best global
fit is a real, uncorrectable inconsistency -- not just floating-point
noise -- and points at a genuinely bad DA3 result for whatever chunk that
node came from, since a bad LOCAL rigid fit (tests/gps.py) would have
silently absorbed it by re-anchoring each segment independently, but one
global fit cannot.

Usage:
    python -m tests.validate_gps_alignment --group g_L18_0
"""
import argparse
import json

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from services.geo import latlon_to_local_m
from street_builder.tab import CLI_JOIN_DATASET_REPO, CLI_RAW_PREFIX, _load_group_meta_pieces
from alignment.gps import fit_similarity_2d


def build_key_to_chunk_map():
    """Every leaf chunk's own pathfind_metadata*.json lists exactly which
    node keys it contributed -- cross-referencing all of them lets us tag
    each node in a merged group with the chunk it actually came from, so
    a bad-residual cluster can be traced back to a specific chunk/merge
    joint instead of just an opaque node key."""
    api = HfApi()
    files = api.list_repo_files(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset")
    meta_files = [f for f in files if f.startswith(CLI_RAW_PREFIX + "/") and "pathfind_metadata" in f and f.endswith(".json")]
    key_to_chunk = {}
    for i, rel in enumerate(meta_files):
        chunk_id = rel.split("/")[1]
        path = hf_hub_download(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset", filename=rel)
        with open(path) as f:
            metadata = json.load(f)
        for key in metadata:
            key_to_chunk[key] = chunk_id
        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(meta_files)} chunk metadata file(s) scanned")
    return key_to_chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", default="/tmp/gps_alignment.json")
    args = parser.parse_args()

    print(f"Loading group {args.group}'s metadata...")
    meta_pieces = _load_group_meta_pieces(args.group)
    print(f"{len(meta_pieces)} piece(s) in group.")

    print("Building node -> chunk_id map from every leaf chunk's raw metadata...")
    key_to_chunk = build_key_to_chunk_map()
    print(f"{len(key_to_chunk)} node(s) attributed to a chunk.")

    results = []
    for piece_idx, (leaf_refs, path_edges, date, reached, frame_poses) in enumerate(meta_pieces):
        keys = list(frame_poses.keys())
        if len(keys) < 3:
            print(f"piece {piece_idx}: only {len(keys)} node(s), skipping (need >=3 for a stable fit)")
            continue

        origin_lat, origin_lon = frame_poses[keys[0]][3], frame_poses[keys[0]][4]
        da3_xz = np.array([[frame_poses[k][0][0], frame_poses[k][0][2]] for k in keys])
        real_en = np.array([latlon_to_local_m(frame_poses[k][3], frame_poses[k][4], origin_lat, origin_lon) for k in keys])

        R, scale, t = fit_similarity_2d(da3_xz, real_en)
        heading = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        fitted = scale * (da3_xz @ R.T) + t
        residuals = np.linalg.norm(fitted - real_en, axis=1)

        print(f"\npiece {piece_idx}: {len(keys)} node(s), scale={scale:.3f}, heading={heading:.1f}deg")
        print(f"  residual: median={np.median(residuals):.1f}m, mean={residuals.mean():.1f}m, "
              f"max={residuals.max():.1f}m, p90={np.percentile(residuals, 90):.1f}m")

        order = np.argsort(-residuals)
        print("  worst 10 node(s):")
        for i in order[:10]:
            k = keys[i]
            print(f"    {k}: residual={residuals[i]:.1f}m  lat/lon=({frame_poses[k][3]:.6f},{frame_poses[k][4]:.6f})  "
                  f"da3_xz=({da3_xz[i][0]:.1f},{da3_xz[i][1]:.1f})")

        for i, k in enumerate(keys):
            results.append({
                "piece": piece_idx, "key": k, "chunk_id": key_to_chunk.get(k),
                "residual_m": float(residuals[i]),
                "lat": frame_poses[k][3], "lon": frame_poses[k][4],
                "da3_xz": da3_xz[i].tolist(),
                "real_en": real_en[i].tolist(), "fitted_en": fitted[i].tolist(),
            })

    with open(args.out, "w") as f:
        json.dump(results, f)
    print(f"\nWrote {len(results)} node result(s) to {args.out}")

    by_chunk = {}
    for r in results:
        by_chunk.setdefault(r["chunk_id"], []).append(r["residual_m"])
    ranked = sorted(by_chunk.items(), key=lambda kv: -np.median(kv[1]))
    print("\nWorst 15 chunk(s) by median residual:")
    for chunk_id, res_list in ranked[:15]:
        print(f"  {chunk_id}: n={len(res_list)}, median={np.median(res_list):.1f}m, max={max(res_list):.1f}m")


if __name__ == "__main__":
    main()
