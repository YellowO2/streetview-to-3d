"""Where the land's triangles have their corners."""
import numpy as np

from streetview_to_3d.postprocess.world.Land import LAND_EVERY
from streetview_to_3d.postprocess.world.land.LandPoints import LandPoints
from streetview_to_3d.postprocess.world.tools.reach import RADIUS_M
from streetview_to_3d.postprocess.world.tools.seams import GAP0_M, GAP_PER, gap_at


def land_corners(site, roads, outlines):
    """LandPoints within RADIUS_M: spaced by their distance from site's cameras (Site), plus
    corners along the roads' and bridges' edges (Roads) and the buildings' walls (Outlines), so
    no triangle spans one."""
    en = _sample_points(RADIUS_M, site.cameras, LAND_EVERY)
    edges = roads.edges()
    walls, owner = _corners(outlines.items, lambda xy: LAND_EVERY * gap_at(site.from_cameras(xy)))
    inside = np.linalg.norm(walls, axis=1) < RADIUS_M
    edges = edges[np.linalg.norm(edges, axis=1) < RADIUS_M]
    owner = np.r_[np.full(len(en) + len(edges), -1), owner[inside]]
    return LandPoints(np.concatenate([en, edges, walls[inside]]), owner)


def _sample_points(radius_m, cams, every=1):
    """(east, north) points within radius_m, every x gap_at(distance to the nearest camera) apart:
    nested jittered grids, each twice the last's spacing, each used where its spacing fits."""
    from scipy.spatial import cKDTree
    tree, rng, out = cKDTree(cams), np.random.default_rng(0), []
    lo_c, hi_c = cams.min(0), cams.max(0)
    s = GAP0_M * every
    while (s / every - GAP0_M) / GAP_PER < 2 * radius_m:
        d_lo, d_hi = (s / every - GAP0_M) / GAP_PER, (2 * s / every - GAP0_M) / GAP_PER
        lo, hi = np.maximum(lo_c - d_hi, -radius_m), np.minimum(hi_c + d_hi, radius_m)
        if (lo < hi).all():
            gx, gy = np.meshgrid(np.arange(lo[0], hi[0], s), np.arange(lo[1], hi[1], s))
            grid = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-0.3 * s, 0.3 * s, (gx.size, 2))
            d = tree.query(grid)[0]
            out.append(grid[(d >= d_lo) & (d < d_hi) & (np.linalg.norm(grid, axis=1) < radius_m)])
        s *= 2
    return np.concatenate(out) if out else np.zeros((0, 2))


def _corners(outlines, spacing):
    """(points (n, 2), owner (n,)) along every outline, spacing(middle) apart: land corners so no triangle spans a wall."""
    import shapely
    pts, owner = [np.zeros((0, 2))], [np.zeros(0, int)]
    for i, (xy, *_) in enumerate(outlines):
        step = float(np.atleast_1d(spacing(xy[:-1].mean(0, keepdims=True)))[0])
        p = shapely.get_coordinates(shapely.segmentize(shapely.linearrings(xy), step))[:-1]
        pts.append(p)
        owner.append(np.full(len(p), i))
    return np.concatenate(pts), np.concatenate(owner)
