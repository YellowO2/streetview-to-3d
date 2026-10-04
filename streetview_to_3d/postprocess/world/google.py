"""Experimental: Google 3D Tiles dabs as the whole world, DA3's points laid onto it. google.ply
(saved by the viewer's ?google-save) is slid onto DA3's points (shift) and gets the panos' look
(seams.unhazed, seams.toward). Each pano's DA3 points are then bent onto Google's shape (bent;
--as-is leaves them), and Google's go only where they stand in front of what a pano saw
(seen_through) and on the water, the map's own, put at Google's height (on_water): behind
DA3's points Google stays, backing them. Rewrites google.ply, water.json, scene.json (the scene is Google's: its map-built
land, roads and buildings left out) and, bent, the nodes' plys.

    python -m streetview_to_3d.postprocess.world.google SCENE_DIR [--as-is]
"""
import json
import os
import sys

import numpy as np
import shapely
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

from streetview_to_3d.common import scene as scene_mod
from streetview_to_3d.postprocess import seams
from streetview_to_3d.postprocess.ply_io import read_node, read_vertices, write_ply
from streetview_to_3d.postprocess.sight import SIGHT_DEG, Sight
from streetview_to_3d.postprocess.world import terrain

FILENAME = "google.ply"
MERGED = "merged"
FIT_REACH_M, FIT_N, FIT_NEAR_M = 15.0, 40_000, 0.3     # shift: Google points this near a pano, about this many
FIT_STEPS_M = 1.0, 0.3, 0.1                            # ... moved by up to 3 of each step in turn
WATER_M, OPEN_M, OPEN_N = 0.5, 10, 200   # on_water: this far over the water's height and all under it; Google's water this far from land, of this many points
LEVEL, WATER_BIN_M = 0.95, 0.1           # ... facing up this much, their heights counted in steps this big
CLEAR_M = 0.1               # seen_through: a Google point this much nearer a pano than its DA3 points behind it
RAYS, RAY_DEG = 5, 1.0      # ... the DA3 points this many, within this of its direction
BEND_DEG = 8.0              # bent: a pano's scale changes this smoothly round the compass
BEND_SAME_M = 1.5           # ... taken from the directions where Google's surface and DA3's are this near: the same one
WALL_DEG = -10.0            # ... looking no lower than this: walls, not the ground
BEND_MIN = 5                # ... of a slice with at least this many of them
BEND_RANGE = 0.85, 1.18     # ... and no more than this
REACH_M = 25.0              # as far as a pano's DA3 points go (clean.blobs.FAR_M)


def shift(pts, scene, cams):
    """The move (x, y, z; m) that lays Google's points onto DA3's: the one putting most of the
    Google points near the panos within FIT_NEAR_M of a DA3 point, searched coarse to fine."""
    near = cKDTree(cams).query(pts, distance_upper_bound=FIT_REACH_M)[0] < np.inf
    sample = pts[near][::max(1, int(near.sum() // FIT_N))]
    tree = cKDTree(scene)
    best = np.zeros(3)
    for step in FIT_STEPS_M:
        moves = best + step * np.stack(np.meshgrid(*[np.arange(-3, 4)] * 3, indexing="ij"), -1).reshape(-1, 3)
        hits = [np.isfinite(tree.query(sample + m, distance_upper_bound=FIT_NEAR_M, workers=-1)[0]).sum()
                for m in moves]
        best = moves[int(np.argmax(hits))]
    return best


def bent(cloud, cam, google):
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


def apart(clouds, cams, pts):
    """Median metres between DA3's surface and Google's, along the panos' lines of sight."""
    tree, out = cKDTree(pts), []
    for cloud, cam in zip(clouds, cams):
        near = tree.query_ball_point(cam, REACH_M)
        if len(cloud) and len(near):
            behind, sure = Sight(cloud, cam).of(pts[near])
            out.append(np.abs(behind[np.abs(behind) <= BEND_SAME_M]))
    return float(np.median(np.concatenate(out))) if out else np.nan


def on_water(pts, normal, scene_dir, name):
    """True for Google points on or under the scene's water (its water.json): within a body's
    outline, off dry land by its shore grid, and no more than WATER_M over the water's height
    (under it: the skirts round Google's tiles) -- the map's water stands there instead, as it
    does for the land. The height is Google's own there: the commonest height of its level
    points well out on the body (OPEN_M from land; boats and bridges are fewer), else the body's
    level; quays and bridges stay. The map's water is put at that height too (water.json's
    levels): its banks are Google's now."""
    out = np.zeros(len(pts), bool)
    if not name or not os.path.exists(os.path.join(scene_dir, name)):
        return out
    with open(os.path.join(scene_dir, name)) as f:
        wet = json.load(f)
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
    with open(os.path.join(scene_dir, name), "w") as f:
        json.dump(wet, f, separators=(",", ":"))
    return out


def seen_through(pts, clouds, cams):
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


def _write_node(scene_dir, node, world):
    """A node's .ply with its points moved to world (every other field kept)."""
    path = os.path.join(scene_dir, node.ply)
    v, header = read_vertices(path)
    T = np.asarray(node.transform, float)
    v = v.copy()
    v["x"], v["y"], v["z"] = ((world - T[:3, 3]) @ np.linalg.inv(T[:3, :3].T)).T
    with open(path, "wb") as f:
        f.write(header.encode("ascii") + v.tobytes())


def merge(scene_dir, log=print, align=True):
    path = os.path.join(scene_dir, FILENAME)
    v, header = read_vertices(path)
    if f"comment {MERGED}\n" in header:
        log("google: already merged")
        return
    pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(float)      # world: y down
    cols = np.stack([v["red"], v["green"], v["blue"]], 1) / 255.0
    gap, normal = v["gap"].astype(float), np.stack([v["nx"], v["ny"], v["nz"]], 1)

    sc = scene_mod.Scene.load(scene_dir)
    scene, scene_cols = terrain.scene_points(sc, scene_dir)
    nodes = [nd for nd in sc.nodes if nd.ply and nd.transform]
    cams = np.array([(np.asarray(nd.transform, float) @ [*nd.position, 1])[:3] for nd in nodes])
    move = shift(pts, scene, cams)
    log(f"google: moved {move.round(2)} m (x east, y down, z north) onto DA3's points")
    pts = pts + move
    xy = pts[:, [0, 2]]

    by_place = cKDTree(xy)

    def google_at(q):
        d, k = by_place.query(q, distance_upper_bound=2 * seams.CELL_M, workers=-1)
        out = np.full((len(q), 3), np.nan)
        out[np.isfinite(d)] = cols[k[np.isfinite(d)]]
        return out
    cols = seams.unhazed(seams.SceneGround.load(scene_dir), google_at)(cols)

    # DA3's points bent onto Google's shape, pano by pano
    clouds = [read_node(scene_dir, nd)[2] for nd in nodes]
    if align:
        was = apart(clouds, cams, pts)
        tree = cKDTree(pts)
        clouds = [bent(c, cam, pts[tree.query_ball_point(cam, REACH_M + BEND_SAME_M)]) for c, cam in zip(clouds, cams)]
        for nd, world in zip(nodes, clouds):
            _write_node(scene_dir, nd, world)
        scene, scene_cols = terrain.scene_points(sc, scene_dir)
        log(f"google: DA3's points bent onto Google's shape: {was:.2f} m apart along the panos' sight before, "
            f"{apart(clouds, cams, pts):.2f} m after")

    # all of Google's stay, taking DA3's colour and look where they meet it, but what stands in
    # front of DA3's points from their pano, and its water
    cols, near = seams.toward(pts, cols, cKDTree(scene) if len(scene) else None, scene_cols)
    air = seen_through(pts, clouds, cams)
    water = on_water(pts, normal, scene_dir, sc.water)
    keep = ~(air | water)

    credit = [l[len("comment "):] for l in header.splitlines() if l.startswith("comment credit")]
    write_ply(path, pts[keep], cols[keep], gap=gap[keep], normal=normal[keep], near=near[keep],
              comments=credit + [MERGED])
    # the scene is Google's now: the viewer draws it for the map's land, roads and buildings
    sc.google, sc.land, sc.terrain, sc.roads, sc.buildings, sc.blocks = FILENAME, None, None, None, None, None
    sc.save(scene_dir)
    log(f"google: {int(keep.sum()):,} of {len(v):,} points kept ({int(air.sum()):,} in front of what a pano "
        f"saw, {int((water & ~air).sum()):,} on the water)")


if __name__ == "__main__":
    merge(sys.argv[1], align="--as-is" not in sys.argv)
