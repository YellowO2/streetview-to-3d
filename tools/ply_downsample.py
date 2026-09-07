"""Random-subsamples a plain (float32 xyz + uint8 rgb) .ply down to a
target fraction of its points, writing a new .ply -- the original is
never touched. Browsers choke well before 100M+ points, so this trims a
big assembled result down to something a Three.js viewer can actually
load.

Usage:
    python -m tools.ply_downsample --in ~/Downloads/ntu_subset_35.ply \
        --out ~/Downloads/ntu_viewer/points.ply --fraction 0.5
"""
import argparse
import os

import numpy as np


def read_ply(path):
    with open(path, "rb") as f:
        data = f.read()
    header_end = data.index(b"end_header\n") + len(b"end_header\n")
    header = data[:header_end].decode("ascii")
    n = int(next(l for l in header.splitlines() if l.startswith("element vertex")).split()[-1])
    has_color = "red" in header
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if has_color:
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    verts = np.frombuffer(data[header_end:], dtype=np.dtype(fields), count=n)
    pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float32)
    cols = (np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.uint8)
            if has_color else None)
    return pts, cols


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
    rgb = cols if cols is not None else np.full((n, 3), 200, dtype=np.uint8)
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fraction", type=float, default=0.5)
    args = parser.parse_args()

    in_path = os.path.expanduser(args.in_path)
    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print(f"Reading {in_path}...")
    pts, cols = read_ply(in_path)
    print(f"{len(pts)} point(s) read.")

    n_keep = int(len(pts) * args.fraction)
    idx = np.random.choice(len(pts), size=n_keep, replace=False)
    pts, cols = pts[idx], (cols[idx] if cols is not None else None)
    print(f"Subsampled to {len(pts)} point(s) ({args.fraction:.0%}).")

    write_ply(out_path, pts, cols)
    print(f"Wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB).")


if __name__ == "__main__":
    main()
