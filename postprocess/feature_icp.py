"""Fit two pieces together in 3D by their road features.

Registering the raw point clouds does not work here, and we spent a long
time finding out why. A road is a big flat sheet, so dense matching lets
one piece slide freely across the other -- every position looks equally
good. The information is not spread over the surface, it is concentrated
in two specific things:

    the road SURFACE   is what fixes height and tilt
    the KERB line      is what fixes sideways position and heading

So this extracts those two things and registers them, rather than
registering everything. Road points are only ever matched to road, kerb
points only ever to kerb -- dense matching would happily pair a kerb
against a random patch of tarmac, which is exactly the sliding we kept
seeing. This is the same idea as feature-based LiDAR registration
(LOAM and friends): pull out edge and surface features, fit on those.

Using the classified kerb rather than a geometric edge matters. Where a
kerb is flush with the grass beside it there is no step to detect, and
anything working from shape alone has nothing to hold on to -- but the
boundary is still perfectly visible, because one side is tarmac and the
other is not.

What this CANNOT do: a straight road with straight kerbs looks identical
at every point along its own length, so nothing here determines how far
along the road a piece sits. That is not a shortcoming of the method, it
is absent from the data. GPS keeps that one. `solve` reports which
directions it actually pinned down, so a caller never has to assume.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from postprocess.road import CELL, road_cells, top_down

SURFACE_BAND_M = 0.75    # points further than this from the road surface are
                         # grass hanging below the kerb, or things standing on
                         # the road -- measured, not guessed: a quarter of the
                         # points inside road cells were foliage below kerb level
KERB_NEIGHBOURS = 8      # points used to estimate the kerb's local direction
ROAD_NEIGHBOURS = 12     # points used to estimate the road's local normal
DEGENERATE_EIGENVALUE = 1e-3   # below this a direction is not constrained at all


def extract_features(piece, bounds, cell=CELL, band=SURFACE_BAND_M, cams=None):
    """(road_xyz, kerb_xyz, paint_xyz) for one piece.

    The kerb comes out as a THIN line -- one point per cell along the
    actual road/not-road boundary. An earlier version dilated the
    boundary and kept every point inside those cells, which turned the
    kerb into a blob covering a quarter of the road. Blobs register
    against blobs at any offset, which defeats the whole point of using
    an edge feature.

    Holes inside the road are filled before taking the boundary. An
    unobserved patch in the middle of the tarmac is a gap in the data,
    not a kerb, and it produced edges running down the centre of the
    road.

    Pass `cams` -- the piece's camera track -- so road_cells can tell the
    road from the footpaths and forecourts beside it. Without it every
    grey surface wide enough to qualify contributes its own boundary, and
    the kerb comes back as many short unrelated fragments."""
    xz, y, cols = piece
    road_mask, _, paint_xz = road_cells(xz, y, cols, bounds, cell, cams=cams)
    _, occ = top_down(xz, y, cols, bounds, cell)

    solid = ndimage.binary_closing(road_mask, np.ones((5, 5)))
    solid = ndimage.binary_fill_holes(solid)
    ring = ndimage.binary_dilation(solid, iterations=1) & ~solid
    kerb_mask = ndimage.binary_dilation(ring & occ, iterations=1) & solid
    # the outer rim, where the panorama ran out of range, is not a kerb
    kerb_mask &= ~(ndimage.binary_dilation(ring & ~occ, iterations=1) & solid)
    kerb_mask &= road_mask

    nx, nz = road_mask.shape
    jx = ((xz[:, 0] - bounds[0]) / cell).astype(int)
    jz = ((xz[:, 1] - bounds[2]) / cell).astype(int)
    inside = (jx >= 0) & (jx < nx) & (jz >= 0) & (jz < nz)
    surf = _surface_height(jx[inside], jz[inside], y[inside], (nx, nz))

    on_surface = np.zeros(len(y), bool)
    on_surface[inside] = np.abs(y[inside] - surf[jx[inside], jz[inside]]) < band

    sel = np.zeros(len(y), bool)
    sel[inside] = road_mask[jx[inside], jz[inside]]
    sel &= on_surface
    road_xyz = np.column_stack([xz[sel, 0], y[sel], xz[sel, 1]])

    # one point per kerb cell, at the cell centre and the road's height there
    kx, kz = np.nonzero(kerb_mask)
    h = surf[kx, kz]
    good = np.isfinite(h)
    kerb_xyz = np.column_stack([bounds[0] + kx[good] * cell, h[good],
                                bounds[2] + kz[good] * cell])

    # Painted markings, given the road's height where each one sits.
    # Worth carrying separately from the kerb: kerbs and lane lines both
    # run ALONG the road and so only ever constrain sideways, but stop
    # bars, give-way lines and junction markings run ACROSS it, and those
    # are the only thing in the scene that can say how far along the road
    # a piece sits.
    px = ((paint_xz[:, 0] - bounds[0]) / cell).astype(int)
    pz = ((paint_xz[:, 1] - bounds[2]) / cell).astype(int)
    ok = (px >= 0) & (px < nx) & (pz >= 0) & (pz < nz)
    px, pz = px[ok], pz[ok]
    ph = surf[px, pz]
    good = np.isfinite(ph)
    paint_xyz = np.column_stack([bounds[0] + px[good] * cell, ph[good],
                                 bounds[2] + pz[good] * cell])
    return road_xyz, kerb_xyz, paint_xyz


def _surface_height(jx, jz, y, shape):
    """Height of the road surface per cell: a rough per-cell median,
    smoothed across neighbours so it tracks the road's slope instead of
    following whatever noise is in one cell."""
    nx, nz = shape
    flat = jx * nz + jz
    order = np.argsort(flat)
    f, yy = flat[order], y[order]
    idx, start = np.unique(f, return_index=True)
    ends = np.append(start[1:], len(f))
    rough = np.full(nx * nz, np.nan)
    for k, (s, e) in enumerate(zip(start, ends)):
        rough[idx[k]] = np.median(yy[s:e])
    rough = rough.reshape(nx, nz)
    filled = np.where(np.isnan(rough), np.nanmedian(rough), rough)
    return ndimage.median_filter(filled, size=9)


def _local_normals(pts, k):
    """Smallest-eigenvector (surface normal) at every point."""
    _, idx = cKDTree(pts).query(pts, k=min(k, len(pts)))
    nb = pts[idx] - pts[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    return np.linalg.eigh(cov)[1][:, :, 0]


def _local_directions(pts, k):
    """Largest-eigenvector (line direction) at every point."""
    _, idx = cKDTree(pts).query(pts, k=min(k, len(pts)))
    nb = pts[idx] - pts[idx].mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", nb, nb)
    return np.linalg.eigh(cov)[1][:, :, 2]


def solve(src_road, src_kerb, dst_road, dst_kerb, iters=25,
          max_pair_dist=1.5, kerb_weight=3.0, max_road_points=20000, seed=0,
          use_road=True, src_paint=None, dst_paint=None, paint_weight=3.0,
          lock_heading=False):
    """3D transform placing src onto dst. Returns (4x4 matrix, report).

    Two kinds of constraint, solved together every iteration:
      road point -> the local PLANE of the road under it   (height, tilt)
      kerb point -> the local LINE of the kerb beside it    (sideways, heading)

    The kerb is weighted up because there are far fewer kerb points than
    road points, and without it the road -- being flat -- would dominate
    and let the fit slide.

    `lock_heading` removes rotation about the VERTICAL axis from the
    system entirely. Tilt and roll are what levelling two road surfaces
    onto a common plane actually needs, and the road constrains them
    well. Heading is different: a straight road looks the same whichever
    end you view it from, so the features barely constrain it, yet a few
    degrees of it swings the far end of a long piece metres sideways --
    left free it reached 12.85 deg and threw piece_5's outermost camera
    14.5 m across the road.

    The degree of freedom is dropped from the least-squares rather than
    solved for and then clamped. Clamping does not constrain the solver,
    it only fights it: the same motion comes back through the tilt axes
    instead, which drove tilt to 45 deg while heading still ended up over
    the limit. Leave it False for a piece whose heading nothing else
    pinned -- a single-node piece genuinely does need to turn freely.
    """
    rng = np.random.default_rng(seed)
    if len(src_road) > max_road_points:
        src_road = src_road[rng.choice(len(src_road), max_road_points, replace=False)]
    if len(dst_road) > max_road_points:
        dst_road = dst_road[rng.choice(len(dst_road), max_road_points, replace=False)]

    if len(dst_road) < 50 or len(dst_kerb) < 20:
        return np.eye(4), {"ok": False, "why": "reference has too few road/kerb features"}

    road_tree, kerb_tree = cKDTree(dst_road), cKDTree(dst_kerb)
    road_n = _local_normals(dst_road, ROAD_NEIGHBOURS)
    kerb_d = _local_directions(dst_kerb, KERB_NEIGHBOURS)
    use_paint = (src_paint is not None and dst_paint is not None
                 and len(src_paint) > 10 and len(dst_paint) > 10)
    if use_paint:
        paint_tree = cKDTree(dst_paint)
        paint_d = _local_directions(dst_paint, KERB_NEIGHBOURS)

    # Rotation is linearised about the features' own centre. Using raw
    # world coordinates here silently breaks the solve: they are hundreds
    # of metres from the origin, so the rotation terms come out hundreds
    # of times larger than the translation ones, translation is left
    # effectively unweighted, and the fit drifts away by tens of metres.
    pivot = np.vstack([src_road, src_kerb]).mean(0)

    T = np.eye(4)
    report = {"ok": True, "iterations": 0, "kerb_pairs": 0, "paint_pairs": 0}
    A = sw = None
    n_kerb = 0
    for it in range(iters):
        road = src_road @ T[:3, :3].T + T[:3, 3]
        kerb = src_kerb @ T[:3, :3].T + T[:3, 3]

        rows, rhs, w = [], [], []
        # road -> plane
        d, i = road_tree.query(road, distance_upper_bound=max_pair_dist)
        ok = np.isfinite(d) & use_road
        if ok.sum() > 20:
            p, q, n = road[ok], dst_road[i[ok]], road_n[i[ok]]
            rows.append(np.hstack([np.cross(p - pivot, n), n]))
            rhs.append(-np.einsum("ij,ij->i", p - q, n))
            w.append(np.ones(ok.sum()))
        # kerb -> line: two constraints per point, perpendicular to the line
        d, i = kerb_tree.query(kerb, distance_upper_bound=max_pair_dist)
        ok = np.isfinite(d)
        n_kerb = int(ok.sum())
        if n_kerb > 10:
            p, q, dirs = kerb[ok], dst_kerb[i[ok]], kerb_d[i[ok]]
            for axis in _perpendiculars(dirs):
                rows.append(np.hstack([np.cross(p - pivot, axis), axis]))
                rhs.append(-np.einsum("ij,ij->i", p - q, axis))
                w.append(np.full(n_kerb, kerb_weight))
        n_paint = 0
        if use_paint:
            paint = src_paint @ T[:3, :3].T + T[:3, 3]
            d, i = paint_tree.query(paint, distance_upper_bound=max_pair_dist)
            ok = np.isfinite(d)
            n_paint = int(ok.sum())
            if n_paint > 10:
                p, q, dirs = paint[ok], dst_paint[i[ok]], paint_d[i[ok]]
                for axis in _perpendiculars(dirs):
                    rows.append(np.hstack([np.cross(p - pivot, axis), axis]))
                    rhs.append(-np.einsum("ij,ij->i", p - q, axis))
                    w.append(np.full(n_paint, paint_weight))
        if not rows:
            report.update(ok=False, why="no feature pairs within range")
            break

        A = np.vstack(rows)
        b = np.concatenate(rhs)
        sw = np.sqrt(np.concatenate(w))
        if lock_heading:
            # column 1 is the rotation about the vertical axis
            Ar = np.delete(A, 1, axis=1)
            xr, *_ = np.linalg.lstsq(Ar * sw[:, None], b * sw, rcond=None)
            x = np.insert(xr, 1, 0.0)
        else:
            x, *_ = np.linalg.lstsq(A * sw[:, None], b * sw, rcond=None)
        T = _increment(x, pivot) @ T
        report["iterations"] = it + 1
        if np.linalg.norm(x[3:]) < 1e-4 and np.linalg.norm(x[:3]) < 1e-5:
            break

    if A is None:
        report.update(ok=False, why="no feature pairs at all",
                      eigenvalues=np.zeros(6), unconstrained=["everything"])
        return np.eye(4), report
    report.update(_conditioning(A, sw))
    report["road_pairs"] = int(np.isfinite(road_tree.query(
        src_road @ T[:3, :3].T + T[:3, 3], distance_upper_bound=max_pair_dist)[0]).sum())
    report["kerb_pairs"] = n_kerb
    report["paint_pairs"] = n_paint
    return T, report


def _perpendiculars(dirs):
    """Two unit vectors perpendicular to each line direction. Distance to
    a line is exactly the distance in these two directions, so a
    point-to-line constraint becomes two plane-like ones -- and a point
    is left free to slide ALONG the kerb, which is correct: a straight
    kerb genuinely cannot say where along itself you are."""
    helper = np.tile([0.0, 1.0, 0.0], (len(dirs), 1))
    flip = np.abs(dirs[:, 1]) > 0.9
    helper[flip] = [1.0, 0.0, 0.0]
    a = np.cross(dirs, helper)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(dirs, a)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return a, b


def _increment(x, pivot):
    """Small rotation about `pivot`, plus translation, as a 4x4."""
    w, t = x[:3], x[3:]
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    R = np.eye(3) + K + K @ K / 2
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pivot + t - R @ pivot
    return T


def _conditioning(A, sw):
    """Which of the six degrees of freedom the features actually pinned.

    A near-zero eigenvalue means the features said nothing at all about
    that direction, and whatever the solver returned for it is arbitrary.
    On a straight road that is normal and expected -- it is how the
    caller knows to keep GPS's answer for sliding along the road."""
    H = (A * sw[:, None]).T @ (A * sw[:, None])
    vals, vecs = np.linalg.eigh(H)
    vals = vals / max(vals.max(), 1e-12)
    weak = []
    for v, vec in zip(vals, vecs.T):
        if v < DEGENERATE_EIGENVALUE:
            rot, trans = np.linalg.norm(vec[:3]), np.linalg.norm(vec[3:])
            kind = "rotation" if rot > trans else "translation"
            axis = vec[:3] if rot > trans else vec[3:]
            weak.append(f"{kind} along ({axis[0]:+.2f},{axis[1]:+.2f},{axis[2]:+.2f})")
    return {"eigenvalues": vals, "unconstrained": weak}


def describe(T, pts):
    """Human-readable summary of what a transform does to `pts`.

    Reports how the points actually move, not the matrix's translation
    column. Those are not the same thing: the rotation happens about a
    point hundreds of metres from the coordinate origin, so the column
    contains a large bookkeeping term and reads as a huge displacement
    even when nothing moved far."""
    R = T[:3, :3]
    yaw = np.degrees(np.arctan2(R[2, 0], R[0, 0]))
    tilt = np.degrees(np.arccos(np.clip(R[1, 1], -1, 1)))
    moved = (pts @ R.T + T[:3, 3]) - pts
    mean = moved.mean(0)
    return (f"moves {np.linalg.norm(mean[[0, 2]]):.2f} m horizontally "
            f"({mean[0]:+.2f}, {mean[2]:+.2f}), height {mean[1]:+.2f} m, "
            f"heading {yaw:+.2f} deg, tilt {tilt:.2f} deg")
