"""Building outlines slid onto DA3's walls of them, and trimmed."""
import numpy as np
from scipy.spatial import cKDTree

from streetview_to_3d.build.postprocess.world.Form import MIN_HEIGHT_M
from streetview_to_3d.build.postprocess.world.Outlines import Outlines
from streetview_to_3d.build.postprocess.world.tools.osm_roads import coverage, lines
from streetview_to_3d.build.postprocess.world.tools.walls import SAME_DEG, near_box, on_plane

FIT_M = 5.0                   # buildings this far past the scene's points are not fitted
WALL_SAMPLE_M = 1.0
SNAP_MAX_M = 2.0              # max slide onto DA3 walls
ROAD_SLACK = 0.02             # a slide may put this much more of an outline onto roads
TRIM_TOL_M, TRIM_FRONT_M = 0.3, 30.0   # trim from this far in front of a DA3 wall, out to this
TRIM_MIN_M2, TRIM_MAX = 1.0, 0.2       # a trim must remove at least this area, at most this share
# DA3's copy of a wall: points within SAME_M of its plane, facing within SAME_DEG, within SAME_NEAR_M,
# at least SAME_MIN; its fitted plane must face within AGREE_DEG; stretches found in SPAN_M steps of
# SPAN_PER_M points per metre, gaps up to BRIDGE_M closed
SAME_M, SAME_NEAR_M, SAME_MIN = 2.5, 5.0, 200
AGREE_DEG = 15
SPAN_M, BRIDGE_M, SPAN_PER_M = 0.2, 0.6, 60
TRIM_EDGE_M = 5.0             # only straight walls this long trim (short curved edges found false planes)
TOP_PERCENTILE = 97


def fitted_outlines(outlines, site, elevation, elements):
    """Outlines of the kept outlines, each slid onto the DA3 walls it has and trimmed to them
    (_fit_to_scene): DA3's points from site (Site), the ground from elevation (Elevation), a
    slide kept off OSM's roads (elements)."""
    if not outlines:
        return Outlines(outlines)
    on_road = coverage(lines(elements, site.to_xy))
    items, fitted, trimmed = _fit_to_scene(outlines, site.points, site.normals, elevation, on_road)
    return Outlines(items, fitted, trimmed)


def _da3_copy(n, d, x, da3, da3_normals, xz):
    """(n2, d2, agrees): plane fitted to DA3's points near wall plane (n, d) through samples x, or None.

    Uses points within SAME_M of the plane, facing within SAME_DEG and within
    SAME_NEAR_M of x (at least SAME_MIN); agrees if within AGREE_DEG of n."""
    lo, hi = x.min(0) - SAME_NEAR_M, x.max(0) + SAME_NEAR_M
    cand = near_box(xz, (lo[[0, 2]] + hi[[0, 2]]) / 2, (hi[[0, 2]] - lo[[0, 2]]) / 2)
    cand = cand[(da3[cand, 1] >= lo[1]) & (da3[cand, 1] <= hi[1])]
    cand = cand[(np.abs(da3[cand] @ n - d) < SAME_M)
                & (np.abs(da3_normals[cand] @ n) > np.cos(np.radians(SAME_DEG)))]
    if len(cand) < SAME_MIN:
        return None
    near, _ = cKDTree(x).query(da3[cand], distance_upper_bound=SAME_NEAR_M)
    P = da3[cand[np.isfinite(near)]]
    if len(P) < SAME_MIN:
        return None
    c = P.mean(0)
    n2 = np.linalg.svd(P - c, full_matrices=False)[2][-1]
    n2 = n2 if n2 @ n > 0 else -n2
    return n2, float(n2 @ c), bool(n2 @ n >= np.cos(np.radians(AGREE_DEG)))


def _stretches(a):
    """[(start, end)] along a wall where DA3 has dense points (SPAN_PER_M per m), gaps to BRIDGE_M closed."""
    steps, count = np.unique(np.floor(a / SPAN_M).astype(int), return_counts=True)
    steps = steps[count >= SPAN_PER_M * SPAN_M]
    if not len(steps):
        return []
    breaks = np.flatnonzero((np.diff(steps) - 1) * SPAN_M >= BRIDGE_M)   # empty steps between
    starts = np.r_[steps[0], steps[breaks + 1]]
    ends = np.r_[steps[breaks], steps[-1]] + 1
    return [(a0 * SPAN_M, a1 * SPAN_M) for a0, a1 in zip(starts, ends)]


def _wall_samples(a, c, base, top):
    """Points every WALL_SAMPLE_M over the wall from a to c, base to top."""
    length = float(np.linalg.norm(c - a))
    u = np.arange(0, length + 1e-9, WALL_SAMPLE_M)
    v = np.arange(base, top + 1e-9, WALL_SAMPLE_M)
    xy = a + (c - a) * (u / max(length, 1e-9))[:, None]
    return np.column_stack([np.repeat(xy[:, 0], len(v)), -np.tile(v, len(u)), np.repeat(xy[:, 1], len(v))])


def _walls(xy, base, h, da3, da3_normals, xz):
    """{edge index: (n2, d2, DA3 point indices on it)} for the walls of xy that DA3 has (da3_copy)."""
    walls = {}
    for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
        length = float(np.linalg.norm(c - a))
        if length < 1e-6:
            continue
        t = (c - a) / length
        n = np.array([t[1], 0.0, -t[0]])                 # outward: outlines run counter-clockwise
        copy = _da3_copy(n, float(n @ [a[0], 0, a[1]]), _wall_samples(a, c, base, base + h), da3, da3_normals, xz)
        if copy is None or not copy[2]:
            continue
        n2, d2, _ = copy
        idx = near_box(xz, (a + c) / 2, length / 2 + FIT_M)
        walls[j] = (n2, d2, idx[on_plane(n2, d2, da3[idx], da3_normals[idx])])
    return walls


def _trim(xy, walls, da3):
    """xy with what stands in front of DA3's walls cut away (DA3's wall is the real face), or None.

    Only in front of the stretches DA3 has, from TRIM_TOL_M out, by walls at
    least TRIM_EDGE_M long. None if the cut is under TRIM_MIN_M2 or over TRIM_MAX."""
    from shapely.geometry import Polygon
    from shapely.geometry.polygon import orient
    from shapely.ops import unary_union
    poly = Polygon(xy).buffer(0)
    strips = []
    for j, (n2, _, on) in walls.items():
        if len(on) < 2 or np.linalg.norm(xy[j + 1] - xy[j]) < TRIM_EDGE_M:
            continue
        n = n2[[0, 2]] / np.linalg.norm(n2[[0, 2]])
        along = np.array([-n[1], n[0]])
        c = da3[on][:, [0, 2]].mean(0)
        for u0, u1 in _stretches((da3[on][:, [0, 2]] - c) @ along):
            near, far = c + n * TRIM_TOL_M, c + n * TRIM_FRONT_M
            strips.append(Polygon([near + along * u0, near + along * u1, far + along * u1, far + along * u0]))
    if not strips:
        return None
    kept = poly.difference(unary_union(strips))
    if kept.geom_type != "Polygon":
        kept = max(getattr(kept, "geoms", []), key=lambda g: g.area, default=kept)
    removed = poly.area - kept.area
    if kept.is_empty or removed < TRIM_MIN_M2 or removed > TRIM_MAX * poly.area:
        return None
    return np.asarray(orient(kept, 1.0).exterior.coords)


def _fit_to_scene(outlines, da3, da3_normals, ground, on_road=None):
    """(outlines, moved, trimmed): buildings slid onto the DA3 walls they have, then trimmed.

    One least-squares slide (no rotation) per building, skipped if over
    SNAP_MAX_M or if it moves the outline onto roads (on_road(xy): share on
    road). A guessed height becomes DA3's wall top (TOP_PERCENTILE). Each
    outline gains {edge: (n2, d2)} for its DA3 walls."""
    if not len(da3):
        return [o + ({},) for o in outlines], 0, 0
    lo, hi = da3[:, [0, 2]].min(0) - FIT_M, da3[:, [0, 2]].max(0) + FIT_M
    xz = cKDTree(da3[:, [0, 2]])
    out, moved, trimmed = [], 0, 0
    for xy, h, guessed, form in outlines:
        if form.part or not ((xy.max(0) >= lo) & (xy.min(0) <= hi)).all():
            out.append((xy, h, guessed, form, {}))
            continue
        base = form.foot(xy, ground)
        walls = _walls(xy, base, h, da3, da3_normals, xz)
        if not walls:
            out.append((xy, h, guessed, form, {}))
            continue
        a = xy[:-1]
        rows = np.array([n2[[0, 2]] for n2, _, _ in walls.values()])
        want = np.array([d2 - n2 @ [a[j][0], 0, a[j][1]] for j, (n2, d2, _) in walls.items()])
        w = np.sqrt([max(len(on), 1) for _, _, on in walls.values()])[:, None]
        shift = np.linalg.lstsq(rows * w, want * w[:, 0], rcond=None)[0]
        if np.linalg.norm(shift) > SNAP_MAX_M or (on_road and on_road(xy + shift) > on_road(xy) + ROAD_SLACK):
            shift = np.zeros(2)
        xy = xy + shift
        moved += bool(shift.any())
        cut = _trim(xy, walls, da3)
        if cut is not None:
            xy, trimmed = cut, trimmed + 1
            walls = _walls(xy, base, h, da3, da3_normals, xz)
        if shift.any() or cut is not None:
            form.roof = form.roof.on(xy)
        if guessed and walls:
            tops = np.concatenate([-da3[on, 1] for _, _, on in walls.values()])
            if len(tops):
                h = max(MIN_HEIGHT_M, np.percentile(tops, TOP_PERCENTILE) - base)
                form.roof.height = min(form.roof.height, h)
        out.append((xy, h, guessed, form, {j: (n2, d2) for j, (n2, d2, _) in walls.items()}))
    return out, moved, trimmed
