"""Reading and writing plain point-cloud .ply files."""
import numpy as np


def read_ply(ply_path):
    """(pts, cols) from a plain binary .ply; cols is None without colour."""
    with open(ply_path, "rb") as f:
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
    cols = (np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float64) / 255.0) if has_color else None
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
    rgb = (np.clip(cols, 0, 1) * 255).astype("u1") if cols is not None else np.full((n, 3), 200, dtype="u1")
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


