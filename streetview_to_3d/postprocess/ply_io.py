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
    kinds = {"float": "<f4", "uchar": "u1", "int": "<i4"}
    vertex = header.split("element vertex")[1].split("element")[0]
    fields = [(l.split()[2], kinds[l.split()[1]]) for l in vertex.splitlines() if l.startswith("property")]
    verts = np.frombuffer(data[header_end:], dtype=np.dtype(fields), count=n)
    pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    cols = (np.stack([verts["red"], verts["green"], verts["blue"]], axis=1).astype(np.float64) / 255.0) if has_color else None
    return pts, cols


def write_ply(path, pts, cols, gap=None, normal=None, kind=None, near=None, sway=None):
    """Points, and with gap (n,) how far each is from its neighbours, metres:
    the viewer draws it that big (scene-store.js, terrainBands). A building's
    also which way each faces (normal (n, 3)), what it is (kind (n,):
    buildings.Blocks) and how near DA3's points it is (near (n,):
    seams.toward). A node's how much each sways in the wind (sway (n,),
    0-1: a tree's, fill), kept as a byte."""
    n = len(pts)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    fields += [("gap", "<f4")] if gap is not None else []
    fields += [("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4")] if normal is not None else []
    fields += [("kind", "u1")] if kind is not None else []
    fields += [("near", "<f4")] if near is not None else []
    fields += [("sway", "u1")] if sway is not None else []
    types = {"<f4": "float", "u1": "uchar"}
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"element vertex {n}\n"
              + "".join(f"property {types[t]} {name}\n" for name, t in fields)
              + "end_header\n").encode("ascii")
    verts = np.zeros(n, dtype=np.dtype(fields))
    if gap is not None:
        verts["gap"] = gap
    if normal is not None:
        verts["nx"], verts["ny"], verts["nz"] = normal[:, 0], normal[:, 1], normal[:, 2]
    if kind is not None:
        verts["kind"] = kind
    if near is not None:
        verts["near"] = near
    if sway is not None:
        verts["sway"] = np.round(np.clip(sway, 0, 1) * 255)
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rgb = (np.clip(cols, 0, 1) * 255).astype("u1") if cols is not None else np.full((n, 3), 200, dtype="u1")
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())


def write_mesh(path, pts, cols, faces, facade=None, gap=None, near=None):
    """A triangle mesh: pts (n, 3), cols (n, 3) 0-1, faces (m, 3) indices
    into them, facade (n, 2) each vertex's place on its wall, metres along
    and up, or (n, 4) with bay/floor sizes, or (n, 7) with its windows' colour too
    (buildings.solid), or None (land); gap (n,) how
    far apart the viewer spaces the points it draws the surface as, there,
    or None; near (n,) how near DA3's points each is (seams.toward), or None."""
    n, m = len(pts), len(faces)
    layout = facade is not None and facade.shape[1] >= 4
    glass = facade is not None and facade.shape[1] >= 7
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        + ("property float facade_u\nproperty float facade_v\n" if facade is not None else "")
        + ("property float facade_bay\nproperty float facade_floor\n" if layout else "")
        + ("property float glass_r\nproperty float glass_g\nproperty float glass_b\n" if glass else "")
        + ("property float gap\n" if gap is not None else "")
        + ("property float near\n" if near is not None else "") +
        f"element face {m}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    fields += [("facade_u", "<f4"), ("facade_v", "<f4")] if facade is not None else []
    fields += [("facade_bay", "<f4"), ("facade_floor", "<f4")] if layout else []
    fields += [("glass_r", "<f4"), ("glass_g", "<f4"), ("glass_b", "<f4")] if glass else []
    fields += [("gap", "<f4")] if gap is not None else []
    fields += [("near", "<f4")] if near is not None else []
    verts = np.zeros(n, dtype=np.dtype(fields))
    verts["x"], verts["y"], verts["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rgb = (np.clip(cols, 0, 1) * 255).astype("u1")
    verts["red"], verts["green"], verts["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    if facade is not None:
        verts["facade_u"], verts["facade_v"] = facade[:, 0], facade[:, 1]
    if layout:
        verts["facade_bay"], verts["facade_floor"] = facade[:, 2], facade[:, 3]
    if glass:
        verts["glass_r"], verts["glass_g"], verts["glass_b"] = facade[:, 4], facade[:, 5], facade[:, 6]
    if gap is not None:
        verts["gap"] = gap
    if near is not None:
        verts["near"] = near
    tris = np.zeros(m, dtype=np.dtype([("n", "u1"), ("i", "<i4", 3)]))
    tris["n"], tris["i"] = 3, faces
    with open(path, "wb") as f:
        f.write(header)
        f.write(verts.tobytes())
        f.write(tris.tobytes())
