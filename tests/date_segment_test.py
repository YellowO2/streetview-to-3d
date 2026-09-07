"""Breaks the reconstruction into per-date segments (all chunks captured
on the same date -- these were reconstructed together and might be
internally correct even if cross-date bridging drifted), fits EACH
date's own nodes to its own GPS positions independently, and reports how
well each date segment lines up on its own. Tests the hypothesis: does
misalignment concentrate at date BOUNDARIES (bridging problem) or is it
scattered within dates too (a different problem)?

Usage:
    python -m tests.date_segment_test --in /tmp/gps_alignment.json
"""
import argparse
import json

import numpy as np
from huggingface_hub import HfApi, hf_hub_download

from street_builder.tab import CLI_JOIN_DATASET_REPO, CLI_RAW_PREFIX
from alignment.gps import fit_similarity_2d


def build_key_to_date():
    api = HfApi()
    files = api.list_repo_files(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset")
    meta_files = [f for f in files if f.startswith(CLI_RAW_PREFIX + "/") and "pathfind_metadata" in f and f.endswith(".json")]
    key_to_date = {}
    for rel in meta_files:
        path = hf_hub_download(repo_id=CLI_JOIN_DATASET_REPO, repo_type="dataset", filename=rel)
        with open(path) as f:
            metadata = json.load(f)
        for key, m in metadata.items():
            key_to_date[key] = m.get("date")
    return key_to_date


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="/tmp/gps_alignment.json")
    parser.add_argument("--out", default="/tmp/gps_date_segments.json")
    args = parser.parse_args()

    with open(args.in_path) as f:
        data = json.load(f)

    print("Building node -> date map from every chunk's raw metadata...")
    key_to_date = build_key_to_date()

    by_date = {}
    for d in data:
        date = key_to_date.get(d["key"])
        if date is None:
            continue
        by_date.setdefault(date, []).append(d)

    print(f"\n{len(by_date)} date(s): {sorted(by_date.keys())}")

    out_nodes = []
    for date, nodes in sorted(by_date.items()):
        if len(nodes) < 3:
            print(f"\n{date}: only {len(nodes)} node(s), skipping (need >=3 for a stable fit)")
            for n in nodes:
                out_nodes.append({**n, "date": date, "fitted_en": n["real_en"]})
            continue
        src = np.array([n["da3_xz"] for n in nodes])
        dst = np.array([n["real_en"] for n in nodes])
        R, scale, t = fit_similarity_2d(src, dst)
        fitted = scale * (src @ R.T) + t
        residuals = np.linalg.norm(fitted - dst, axis=1)
        print(f"\n{date}: {len(nodes)} node(s), scale={scale:.3f}, "
              f"median residual={np.median(residuals):.1f}m, max={residuals.max():.1f}m, "
              f"chunks={sorted({n['chunk_id'] for n in nodes})}")
        for n, f, r in zip(nodes, fitted, residuals):
            out_nodes.append({**n, "date": date, "fitted_en": f.tolist(), "residual_m": float(r)})

    with open(args.out, "w") as f:
        json.dump(out_nodes, f)
    print(f"\nWrote {len(out_nodes)} node result(s) to {args.out}")


if __name__ == "__main__":
    main()
