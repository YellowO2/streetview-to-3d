"""The land round the buildings shaped so they stand on it, and they on it."""
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

BESIDE_M = 1.5                # DA3 ground this near a corner counts as beside it
PAD_M, CUT_SLOPE = 1.0, 0.5   # land kept level this far round a building's foot, then rising at most this
CUT_REACH_M = 20.0            # ... out to this far


def stood_buildings(points, outlines, site):
    """LandPoints with the land raised to DA3's ground beside buildings (site: Site, its ground),
    then cut to each building's foot; each outline's Form gets its foot and skirt (Outlines)."""
    under = points.dist == 0
    h = _onto_scene(outlines.items, points.xy, points.height, points.owner, site.ground, under)
    h = _settle(outlines.items, points.xy, h, points.owner)
    return replace(points, height=h)


def _onto_scene(outlines, xy, h, owner, own, under):
    """Land heights h raised to DA3's ground round buildings with DA3 ground beside two or more corners.

    Corners within BESIDE_M of own (postprocess/SceneGround.py) take its height; the
    land round them rises to the median, falling off at CUT_SLOPE past PAD_M.
    Never where under (DA3's own ground) is true."""
    import shapely
    if not len(owner):
        return h
    dist, ground, _ = own.at(xy)
    tree, out = cKDTree(xy), h.copy()
    for i, (ring, _, _, form, *_) in enumerate(outlines):
        mine = np.flatnonzero(owner == i)
        if not len(mine) or form.base_m > 0:
            continue
        has = mine[dist[mine] <= BESIDE_M]
        if len(has) < 2:
            continue
        out[has] = ground[has]
        level = float(np.median(ground[has]))
        c = ring[:-1].mean(0)
        reach = np.linalg.norm(ring - c, axis=1).max() + PAD_M + CUT_REACH_M
        idx = np.asarray(tree.query_ball_point(c, reach), int)
        idx = idx[~np.isin(idx, has) & ~under[idx]]
        if not len(idx):
            continue
        d = shapely.distance(shapely.polygons(ring), shapely.points(xy[idx]))
        out[idx] = np.maximum(out[idx], level - np.maximum(d - PAD_M, 0) * CUT_SLOPE)
    return out


def _settle(outlines, xy, h, owner):
    """Stand every outline on the land and cut the land to it; returns the new h.

    Foot (Form.foot_m) is the lowest land height at its corners; land within
    PAD_M is cut to it, rising at most CUT_SLOPE beyond (only ever lowered).
    Where a neighbour's cut goes lower, walls extend down (Form.skirt_m)."""
    import shapely
    if not len(owner):
        return h
    tree, out = cKDTree(xy), h.copy()
    for i, (ring, _, _, form, *_) in enumerate(outlines):
        mine = owner == i
        if mine.any():
            form.foot_m = float(h[mine].min())
    for _, _, _, form, *_ in outlines:
        if form.stand_on is not None and form.stand_on.foot_m is not None:
            form.foot_m = form.stand_on.foot_m
    for ring, _, _, form, *_ in outlines:
        if form.foot_m is None:
            continue
        c = ring[:-1].mean(0)
        reach = np.linalg.norm(ring - c, axis=1).max() + PAD_M + CUT_REACH_M
        idx = np.asarray(tree.query_ball_point(c, reach), int)
        if not len(idx):
            continue
        d = shapely.distance(shapely.polygons(ring), shapely.points(xy[idx]))
        out[idx] = np.minimum(out[idx], form.foot_m + np.maximum(d - PAD_M, 0) * CUT_SLOPE)
    for i, (_, _, _, form, *_) in enumerate(outlines):
        mine = owner == i
        if mine.any():
            form.skirt_m = max(0.0, form.foot_m - float(out[mine].min()))
    return out
