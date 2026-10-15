"""Building points DA3 already covers left to it, the rest faded into it."""
import numpy as np

from streetview_to_3d.build.postprocess.world.buildings.Blocks import ROOF
from streetview_to_3d.build.postprocess.world.tools.seams import COVER, ramp
from streetview_to_3d.build.postprocess.world.tools.walls import near_box, on_plane

PULL_MAX_M = 1.0              # max pull of seam points onto DA3's plane
SEAM_FADE_M = 3.0             # seam width
SEAM_TINT, SEAM_MIN = 0.8, 20


def seam(blocks, site):
    """blocks (Blocks) of the buildings DA3 reaches only, without the points DA3's (site: Site)
    already cover there, the rest faded into them (_seam); cut: how many went."""
    blocks.take(blocks.reached[blocks.which])
    if not len(blocks.pts):
        return blocks
    # leave to DA3 what it already has of each building
    roofs_near = np.full(len(blocks.pts), np.inf)                 # only roofs use it
    roof = blocks.edge == ROOF
    if site.tree is not None and roof.any():
        roofs_near[roof] = site.tree.query(blocks.pts[roof], distance_upper_bound=1.0, workers=-1)[0]
    blocks.cut = _seam(blocks, site.points, site.normals, site.colours, roofs_near)
    return blocks


def _seam(blocks, da3, da3_normals, da3_cols, roofs_near):
    """Drop building points DA3 already covers and fade the rest into DA3; returns how many were dropped.

    On walls with a DA3 plane, a point within COVER x its gap of a DA3 point
    on the plane (in wall coordinates) goes; within SEAM_FADE_M the rest are
    pulled onto the plane (at most PULL_MAX_M) and tinted toward DA3. Roof
    points go by roofs_near, their distance to DA3."""
    from scipy.spatial import cKDTree
    keep = np.ones(len(blocks.pts), bool)
    roof = blocks.edge == ROOF
    keep[roof] = roofs_near[roof] > COVER * blocks.gap[roof]
    order = np.argsort(blocks.edge, kind="stable")
    bounds = np.searchsorted(blocks.edge[order], np.arange(len(blocks.edges) + 1))
    xz = cKDTree(da3[:, [0, 2]]) if len(da3) else None
    for e, (a, t, length, plane) in enumerate(blocks.edges):
        mine = order[bounds[e]:bounds[e + 1]]
        if plane is None or not len(mine) or xz is None:
            continue
        n2, d2 = plane
        idx = near_box(xz, a + t * length / 2, length / 2 + SEAM_FADE_M + 1)
        idx = idx[on_plane(n2, d2, da3[idx], da3_normals[idx])]
        if len(idx) < SEAM_MIN:
            continue
        u = (da3[idx][:, [0, 2]] - a) @ t
        d, k = cKDTree(np.c_[u, -da3[idx, 1]]).query(np.c_[blocks.u[mine], blocks.v[mine]])
        band = ramp(d / SEAM_FADE_M)
        keep[mine] = d > COVER * blocks.gap[mine]
        off = np.clip(d2 - blocks.pts[mine] @ n2, -PULL_MAX_M, PULL_MAX_M)
        blocks.pts[mine] += ((1 - band) * off)[:, None] * n2
        w = (SEAM_TINT * (1 - band))[:, None]
        blocks.cols[mine] = blocks.cols[mine] * (1 - w) + da3_cols[idx[k]] * w
    blocks.take(keep)
    return int(len(keep) - keep.sum())
