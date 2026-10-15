"""The world with Google's points standing for the map-built one."""
import copy
import logging
from dataclasses import replace

import numpy as np
import shapely
from scipy.spatial import cKDTree

from streetview_to_3d.postprocess.world.google.GooglePoints import MERGED
from streetview_to_3d.postprocess.world.tools.file_names import GOOGLE, WATER
from streetview_to_3d.postprocess.world.tools.scene_points import scene_points
from streetview_to_3d.postprocess.world.tools.seams import toward

logger = logging.getLogger(__name__)

WATER_M, OPEN_M, OPEN_N = 0.5, 10, 200   # _on_water: this far over the water's height and all under it; Google's water this far from land, of this many points
LEVEL, WATER_BIN_M = 0.95, 0.1           # ... facing up this much, their heights counted in steps this big
CLEAR_M = 0.1               # _seen_through: a Google point this much nearer a pano than its DA3 points behind it
RAYS, RAY_DEG = 5, 1.0      # ... the DA3 points this many, within this of its direction


def merged_world(points, scene, clouds, centres, world):
    """world (World) with Google's points (GooglePoints, laid on the scene) as google.ply: all of
    them, taking DA3's colour and look where they meet it (toward), but those standing in front of
    what a pano saw (_seen_through) and those on the water (_on_water), whose levels become
    Google's. clouds, centres: each placed pano's DA3 points in the world and its camera.
    The world as it is if the points were merged before."""
    if points.merged:
        return world
    scene_, scene_cols = scene_points(scene)
    cols, near = toward(points.pts, points.cols, cKDTree(scene_) if len(scene_) else None, scene_cols)
    air = _seen_through(points.pts, clouds, centres)
    wet = copy.deepcopy(world.json.get(WATER))
    water = _on_water(points.pts, points.normal, wet)
    keep = ~(air | water)
    google = dict(pts=points.pts[keep], cols=cols[keep], gap=points.gap[keep], normal=points.normal[keep],
                  near=near[keep], comments=points.credit + [MERGED])
    json = dict(world.json, **({WATER: wet} if wet is not None else {}))
    logger.info(f"google: {int(keep.sum()):,} of {len(points.pts):,} points kept ({int(air.sum()):,} in front of what a pano "
                f"saw, {int((water & ~air).sum()):,} on the water)")
    return replace(world, points=dict(world.points, **{GOOGLE: google}), json=json)


def _on_water(pts, normal, wet):
    """True for Google points on or under the scene's water (wet: water.json's contents): within a body's
    outline, off dry land by its shore grid, and no more than WATER_M over the water's height
    (under it: the skirts round Google's tiles) -- the map's water stands there instead, as it
    does for the land. The height is Google's own there: the commonest height of its level
    points well out on the body (OPEN_M from land; boats and bridges are fewer), else the body's
    level; quays and bridges stay. The map's water is put at that height too (wet's levels, set in
    place): its banks are Google's now."""
    out = np.zeros(len(pts), bool)
    if wet is None:
        return out
    shore = wet["shore"]
    grid = np.asarray(shore["metres"]).reshape(shore["size"], shore["size"])
    xy, up = pts[:, [0, 2]], -pts[:, 1]
    c = np.floor((xy - shore["lo"]) / shore["cell"]).astype(int)
    ok = ((c >= 0) & (c < shore["size"])).all(1)
    from_land = np.zeros(len(pts))
    from_land[ok] = grid[c[ok, 1], c[ok, 0]]
    for body in wet["surfaces"]:
        at = np.flatnonzero(from_land > 0)
        at = at[shapely.contains_xy(shapely.Polygon(body["outer"], body["holes"]), *xy[at].T)]
        open_ = up[at][(from_land[at] >= OPEN_M) & (np.abs(normal[at, 1]) > LEVEL)]
        if len(open_) >= OPEN_N:
            n, edges = np.histogram(open_, np.arange(open_.min(), open_.max() + 2 * WATER_BIN_M, WATER_BIN_M))
            body["level"] = round(float(edges[n.argmax()] + WATER_BIN_M / 2), 2)
        out[at[up[at] < body["level"] + WATER_M]] = True
    return out


def _seen_through(pts, clouds, cams):
    """True for Google points a pano looked through: standing in front of its DA3 points. From
    the pano, one of the RAYS DA3 points nearest the Google point's direction (within RAY_DEG
    of it) lies over CLEAR_M further than it."""
    gone = np.zeros(len(pts), bool)
    tree = cKDTree(pts)
    for cloud, cam in zip(clouds, cams):
        if not len(cloud):
            continue
        rel = cloud - cam
        depth = np.linalg.norm(rel, axis=1)
        near = np.asarray(tree.query_ball_point(cam, depth.max()))
        near = near[~gone[near]] if len(near) else near
        if not len(near):
            continue
        to = pts[near] - cam
        far = np.linalg.norm(to, axis=1)
        d, k = cKDTree(rel / depth[:, None]).query(to / far[:, None], k=RAYS,
                                                   distance_upper_bound=2 * np.sin(np.radians(RAY_DEG) / 2), workers=-1)
        behind = depth[np.minimum(k, len(depth) - 1)] - far[:, None]      # each ray's DA3 point past the Google point
        gone[near] = (np.isfinite(d) & (behind > CLEAR_M)).any(1)
    return gone
