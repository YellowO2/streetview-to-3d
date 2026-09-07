"""Converts a plain (float32 xyz + uint8 rgb) .ply into a proper .las file
for PotreeConverter -- fixes two things a generic `pdal translate` gets
wrong for our data:

1. Up-axis: our reconstruction pipeline outputs Y-up (small Y range =
   building heights), but LAS/Potree assume Z-up. Remapped via the
   standard Y-up -> Z-up rotation (new_x=x, new_y=-z, new_z=y), which
   preserves right-handedness (no mirroring).
2. Color depth: our .ply stores 8-bit (0-255) RGB, but LAS RGB fields are
   16-bit (0-65535) by convention -- writing 0-255 straight into them
   reads as near-black to any standard LAS viewer. Scaled by 257 (so 255
   -> 65535) to fill the full range.

Usage:
    python -m tests.ply_to_las --in ~/Downloads/ntu_subset_35.ply --out ~/Downloads/ntu_subset_35.las
"""
import argparse
import os

import laspy
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
    pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    cols = (np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.uint16)
            if has_color else None)
    return pts, cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    in_path = os.path.expanduser(args.in_path)
    out_path = os.path.expanduser(args.out)

    print(f"Reading {in_path}...")
    pts, cols = read_ply(in_path)
    print(f"{len(pts)} point(s) read.")

    # Y-up -> Z-up, right-handed (no mirroring): x stays, old -z becomes
    # new y, old y becomes new z.
    remapped = np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1)

    header = laspy.LasHeader(point_format=7, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = remapped.min(axis=0)

    las = laspy.LasData(header)
    las.x, las.y, las.z = remapped[:, 0], remapped[:, 1], remapped[:, 2]
    if cols is not None:
        # 8-bit (0-255) -> 16-bit (0-65535): scale by 257 so 255 -> 65535.
        las.red, las.green, las.blue = cols[:, 0] * 257, cols[:, 1] * 257, cols[:, 2] * 257

    print(f"Writing {out_path}...")
    las.write(out_path)
    print(f"Wrote {out_path} ({os.path.getsize(out_path)/1e6:.1f} MB).")


if __name__ == "__main__":
    main()
