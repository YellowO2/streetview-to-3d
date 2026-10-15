"""Each pano's DA3 points bent onto the shape of Google's."""
import logging
from dataclasses import replace

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from streetview_to_3d.build.postprocess.Sight import SIGHT_DEG, Sight

logger = logging.getLogger(__name__)

BEND_DEG = 8.0              # _bent: a pano's scale changes this smoothly round the compass
BEND_SAME_M = 1.5           # ... taken from the directions where Google's surface and DA3's are this near: the same one
WALL_DEG = -10.0            # ... looking no lower than this: walls, not the ground
BEND_MIN = 5                # ... of a slice with at least this many of them
BEND_RANGE = 0.85, 1.18     # ... and no more than this
REACH_M = 25.0              # as far as a pano's DA3 points go (clean.drop_blobs.FAR_M)


def bent(scene, centres, points, settings):
    """(scene, clouds): each placed pano's DA3 points in the world (clouds), bent onto Google's
    shape (points: GooglePoints, laid on DA3's) and put back in the scene's nodes; as they are
    if settings.bend is off (or the points were merged before). centres: pano_centres."""
    nodes = scene.placed()
    clouds = [nd.cloud.world(nd.transform) for nd in nodes]
    if not settings.bend or points.merged:
        return scene, clouds
    pts = points.pts
    was = _apart(clouds, centres, pts)
    tree = cKDTree(pts)
    clouds = [_bent(c, cam, pts[tree.query_ball_point(cam, REACH_M + BEND_SAME_M)]) for c, cam in zip(clouds, centres)]
    for nd, world in zip(nodes, clouds):
        T = np.asarray(nd.transform, float)
        nd.cloud = replace(nd.cloud, points=(world - T[:3, 3]) @ np.linalg.inv(T[:3, :3].T))
    logger.info(f"google: DA3's points bent onto Google's shape: {was:.2f} m apart along the panos' sight before, "
                f"{_apart(clouds, centres, pts):.2f} m after")
    return scene, clouds


def _bent(cloud, cam, google):
    """A pano's DA3 points (cloud, seen from cam) bent onto Google's (google: those about it).

    DA3's sheet is right in what it shows and a little off in shape, as paper bent: nearer or
    further than it should be, differently round the compass. Its view in upright slices
    (sight's directions, ground to sky), each is stretched from the pano as one, level (heights
    kept: the ground is right already) -- by how much further Google's surface is than DA3's
    where they show the same one above the ground (the slice's median over WALL_DEG up),
    smoothed round the circle over BEND_DEG, a slice with too few taking its neighbours'. What
    stands in a slice stays straight and together."""
    if not len(cloud) or not len(google):
        return cloud
    da3, there = Sight(cloud, cam), Sight(google, cam)
    with np.errstate(invalid="ignore"):
        same = np.isfinite(da3.saw) & np.isfinite(there.saw) & (np.abs(there.saw - da3.saw) <= BEND_SAME_M)
        ratio = there.saw / da3.saw
    same[:, (np.arange(da3.n[1]) + .5) * SIGHT_DEG - 90 < WALL_DEG] = False      # walls, not the ground
    slices = np.array([np.median(r[m]) if m.sum() >= BEND_MIN else np.nan for r, m in zip(ratio, same)])
    known = np.isfinite(slices)
    if not known.any():
        return cloud
    blur = lambda a: gaussian_filter1d(a, BEND_DEG / SIGHT_DEG, mode="wrap")
    share = blur(known.astype(float))
    by = np.where(share > 1e-3, blur(np.where(known, np.clip(slices, *BEND_RANGE), 0)) / np.maximum(share, 1e-3),
                  np.median(slices[known]))
    # each point by its own bearing, between its two slices' scales
    rel = cloud - cam
    at = (np.arctan2(rel[:, 0], rel[:, 2]) + np.pi) / (2 * np.pi) * len(by) - .5
    lo, t = np.floor(at).astype(int), at - np.floor(at)
    rel[:, [0, 2]] *= ((1 - t) * by[lo % len(by)] + t * by[(lo + 1) % len(by)])[:, None]
    return cam + rel


def _apart(clouds, cams, pts):
    """Median metres between DA3's surface and Google's, along the panos' lines of sight."""
    tree, out = cKDTree(pts), []
    for cloud, cam in zip(clouds, cams):
        near = tree.query_ball_point(cam, REACH_M)
        if len(cloud) and len(near):
            behind, sure = Sight(cloud, cam).of(pts[near])
            out.append(np.abs(behind[np.abs(behind) <= BEND_SAME_M]))
    return float(np.median(np.concatenate(out))) if out else np.nan
