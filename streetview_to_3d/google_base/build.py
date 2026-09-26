"""Build the Google base: one clean, flat-surfaced shape of the street from
Google's depth maps. Shape only -- colour comes later, from painting.

1. Each pano on its own planes. A Google depth map is a list of planes plus
   a plane number per pixel, so every ray is put exactly on its own plane:
   no noise, no stair-steps. Only the dense part of each pano is kept --
   where one depth pixel covers under MAX_SPACING_M of surface; far or
   edge-on surfaces are a guess. Tiny planes (under PLANE_MIN_PX pixels)
   are dropped.
2. Walls merged across panos. Planes of different panos that are the same
   surface (within MERGE_DEG, MERGE_M, footprints overlapping by OVERLAP)
   become one plane, fitted to all their dense points together, and their
   rays are re-cast onto it. Floors take no part: moving some of a floor's
   planes and not their neighbours tears it, and re-casting a steep plane's
   rays onto a floor spreads them into sparse rings.
3. One ground. Every pano's ground (postprocess.ground) goes into one height
   map: each 0.5 m square takes the ground of the pano whose camera is
   closest (it sees that spot best), lightly smoothed. The ground is then
   rebuilt as an even GROUND_STEP_M grid on that map, only where some pano
   saw ground -- one surface, no stacked floors, no rings from depth rows.

Panos stay where their GPS, elevation and heading put them: nudging them
to agree with each other was tried and made walls worse.
"""
import time
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from streetview_to_3d.postprocess.ground import ground

GRID_W = 1024                  # rays per pano: GRID_W x GRID_W/2
MAX_M = 50.0                   # a backstop only; density decides where surfaces stop
PLANE_MIN_PX = 30              # smaller planes (native pixels) are edge junk
PIXEL_RAD = 2 * np.pi / 512    # one native depth pixel
MAX_SPACING_M = 0.75           # keep where one depth pixel covers less surface than this
MERGE_DEG, MERGE_M, OVERLAP, FOOT_M = 30.0, 0.4, 0.3, 0.5
FLOOR_UP_DEG = 20              # planes facing up within this are floors (step 3's business)
GROUND_CELL_M, GROUND_SMOOTH_M, GROUND_STEP_M = 0.5, 0.5, 0.05


def rays(h, w):
    """Unit rays of an h x w equirect grid in the photo frame (x right, y
    down, z forward), flattened, with each pixel's (u, v) in [0, 1)."""
    y, x = np.mgrid[0:h, 0:w]
    u, v = ((x + .5) / w).ravel(), ((y + .5) / h).ravel()
    lon, lat = (u - .5) * 2 * np.pi, (v - .5) * np.pi
    return np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1), u, v


def own_planes(p):
    """(label, planes) of pano p: label is each native pixel's plane (-1 =
    none), planes is (normal, distance, that plane's points) in the photo
    frame, n . x = distance, fitted exactly to its pixels."""
    h, w = p.depth.shape
    r, _, _ = rays(h, w)
    pts = (r * p.depth.reshape(-1, 1)).reshape(h, w, 3)
    num = np.where(p.depth > 0, p.plane_index, 0)
    label = np.full((h, w), -1)
    planes = []
    for g in np.unique(num):
        m = num == g
        if g == 0 or m.sum() < PLANE_MIN_PX:
            continue
        label[m] = len(planes)
        q = pts[m]
        c = q.mean(0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
        if n @ c < 0:
            n = -n
        planes.append((n, float(n @ c), q))
    return label, planes


def dense(q, n):
    """Which points q (photo frame) on a plane with normal n are densely
    seen: one depth pixel there covers under MAX_SPACING_M of surface."""
    r = np.linalg.norm(q, axis=1)
    facing = np.abs(q @ n) / np.maximum(r, 1e-9)
    return r * PIXEL_RAD / np.maximum(facing, 1e-3) < MAX_SPACING_M


def _footprint(x, n, d):
    """0.5 m cells of points x laid flat onto the plane (n, d)."""
    c = np.floor((x - np.outer(x @ n - d, n)) / FOOT_M).astype(np.int64) + 2 ** 20
    return np.unique((c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2])


def group_planes(entries):
    """Merge planes of different panos that are the same surface.

    entries: dicts with world normal n, distance d (n . x = d), dense world
    points x and pano k. Returns each entry's final (n, d). Biggest planes
    go first; a plane joins the first group within MERGE_DEG and MERGE_M
    whose footprint it overlaps by OVERLAP, and the group's plane is refitted
    to all its members' points. Floors keep their own plane and take no members.
    """
    order = sorted(range(len(entries)), key=lambda i: -len(entries[i]["x"]))
    groups, of = [], [None] * len(entries)
    floor_cos, merge_cos = np.cos(np.radians(FLOOR_UP_DEG)), np.cos(np.radians(MERGE_DEG))
    for i in order:
        a = entries[i]
        if abs(a["n"][1]) > floor_cos:
            groups.append(dict(members=[i], n=a["n"], d=a["d"], floor=True))
            of[i] = groups[-1]
            continue
        best = None
        for g in groups:
            if g["floor"] or a["n"] @ g["n"] < merge_cos or abs(a["d"] - g["d"]) > MERGE_M:
                continue
            fa = _footprint(a["x"], g["n"], g["d"])
            if len(np.intersect1d(fa, g["foot"], assume_unique=True)) >= OVERLAP * min(len(fa), len(g["foot"])):
                best = g
                break
        if best is None:
            groups.append(dict(members=[i], n=a["n"], d=a["d"], floor=False,
                               foot=_footprint(a["x"], a["n"], a["d"])))
            of[i] = groups[-1]
            continue
        best["members"].append(i)
        X = np.concatenate([entries[m]["x"] for m in best["members"]])
        c = X.mean(0)
        n = np.linalg.svd(X - c, full_matrices=False)[2][-1]
        best["n"], best["d"] = (n, float(n @ c)) if n @ best["n"] > 0 else (-n, float(-n @ c))
        best["foot"] = np.union1d(best["foot"], _footprint(a["x"], best["n"], best["d"]))
        of[i] = best
    shared = sum(len({entries[m]["k"] for m in g["members"]}) > 1 for g in groups)
    return [(g["n"], g["d"]) for g in of], len(groups), shared


def shared_ground(xs, cams):
    """One ground from every pano's ground points xs[k] (world): each
    GROUND_CELL_M square takes the median height of the pano whose camera
    (cams[k]) is closest, the map is lightly smoothed, and the ground is laid
    out as an even GROUND_STEP_M grid over the squares that had any.
    Returns (points, owner pano of each point)."""
    who = np.concatenate([np.full(len(x), k) for k, x in enumerate(xs)])
    X = np.concatenate(xs)
    if not len(X):
        return np.zeros((0, 3)), np.zeros(0, int)
    lo = X[:, [0, 2]].min(0) - 2 * GROUND_CELL_M
    ij = np.floor((X[:, [0, 2]] - lo) / GROUND_CELL_M).astype(int)
    dims = ij.max(0) + 3
    centre = (ij + .5) * GROUND_CELL_M + lo
    dcam = np.hypot(centre[:, 0] - cams[who, 0], centre[:, 1] - cams[who, 2])
    flat = ij[:, 0] * dims[1] + ij[:, 1]
    order = np.lexsort((dcam, flat))
    fs = flat[order]
    first = np.r_[True, fs[1:] != fs[:-1]]
    win = np.full(dims.prod(), -1)
    win[fs[first]] = who[order][first]
    mine = win[flat] == who
    o2 = np.argsort(flat[mine])
    keys, start = np.unique(flat[mine][o2], return_index=True)
    H = np.full(dims.prod(), np.nan)
    H[keys] = [np.median(s) for s in np.split(X[mine][o2, 1], start[1:])]
    H = H.reshape(dims)
    have = np.isfinite(H)
    sigma = GROUND_SMOOTH_M / GROUND_CELL_M
    num = gaussian_filter(np.where(have, H, 0), sigma)
    den = gaussian_filter(have.astype(float), sigma)
    Hs = np.where(den > 1e-3, num / np.maximum(den, 1e-9), np.nan)

    per = int(round(GROUND_CELL_M / GROUND_STEP_M))
    sq = np.argwhere(have)
    off = (np.arange(per) + .5) * GROUND_STEP_M
    ox, oz = np.meshgrid(off, off, indexing="ij")
    xz = (lo + sq[:, None, :] * GROUND_CELL_M + np.stack([ox.ravel(), oz.ravel()], 1)[None]).reshape(-1, 2)
    owner = np.repeat(win[sq[:, 0] * dims[1] + sq[:, 1]], per * per)
    # bilinear on the smoothed map, empty corners left out
    g = (xz - lo) / GROUND_CELL_M - .5
    i0 = np.clip(np.floor(g).astype(int), 0, dims - 2)
    f = g - i0
    val = np.zeros(len(xz))
    wsum = np.zeros(len(xz))
    for a in (0, 1):
        for b in (0, 1):
            c = Hs[i0[:, 0] + a, i0[:, 1] + b]
            w = (f[:, 0] if a else 1 - f[:, 0]) * (f[:, 1] if b else 1 - f[:, 1])
            w = np.where(np.isfinite(c), w, 0)
            val += w * np.nan_to_num(c)
            wsum += w
    ok = wsum > 0
    y = val[ok] / wsum[ok]
    return np.stack([xz[ok, 0], y, xz[ok, 1]], 1), owner[ok]


@dataclass
class Base:
    """The Google base in the scene's frame: each pano's surfaces except the
    ground (points[k]), and the one shared ground."""
    panos: list
    points: list
    ground: np.ndarray
    ground_owner: np.ndarray


def build(panos, log=print):
    """The Google base from placed GooglePanos (google_base.fetch)."""
    t0 = time.time()
    R_, U, V = rays(GRID_W // 2, GRID_W)
    # 1. each pano on its own planes, dense part only
    entries, per_pano = [], []
    for k, p in enumerate(panos):
        label, planes = own_planes(p)
        h, w = label.shape
        ray_label = label[np.clip((V * h).astype(int), 0, h - 1), np.clip((U * w).astype(int), 0, w - 1)]
        d = np.full(len(U), np.nan)
        for j, (n, dist, q) in enumerate(planes):
            m = ray_label == j
            den = R_[m] @ n
            d[m] = dist / np.where(np.abs(den) > 0.05, den, np.nan)
            nw = n @ p.R
            x = p.pos + q[dense(q, n)] @ p.R
            entries.append(dict(k=k, j=j, n=nw, d=dist + nw @ p.pos, x=x if len(x) else p.pos + q @ p.R))
        # each ray's density, judged on its own plane
        nrm = np.zeros((len(U), 3))
        for j, (n, _, _) in enumerate(planes):
            nrm[ray_label == j] = n
        facing = np.abs((R_ * nrm).sum(1))
        keep = np.isfinite(d) & (d > 0) & (d < MAX_M) & (np.nan_to_num(d) * PIXEL_RAD / np.maximum(facing, 1e-3) < MAX_SPACING_M)
        per_pano.append((ray_label, keep))
    t1 = time.time()

    # 2. walls merged across panos, rays re-cast onto the merged planes
    final, n_surfaces, shared = group_planes(entries)
    lookup = {(e["k"], e["j"]): nd for e, nd in zip(entries, final)}
    xs, ns = [], []
    for k, p in enumerate(panos):
        ray_label, keep = per_pano[k]
        dirs = R_ @ p.R
        t = np.full(len(U), np.nan)
        nw = np.zeros((len(U), 3))
        for j in np.unique(ray_label[ray_label >= 0]):
            n, dist = lookup[(k, j)]
            m = ray_label == j
            nw[m] = n
            den = dirs[m] @ n
            tt = (dist - n @ p.pos) / np.where(np.abs(den) > 0.05, den, np.nan)
            t[m] = np.where((tt > 0) & (tt < MAX_M), tt, np.nan)
        ok = np.isfinite(t) & keep
        xs.append(p.pos + dirs[ok] * t[ok, None])
        ns.append(nw[ok])
    t2 = time.time()

    # 3. one shared ground, rebuilt as an even grid
    gmask = [ground(x, p.pos, normals=n) for x, n, p in zip(xs, ns, panos)]
    G, owner = shared_ground([x[m] for x, m in zip(xs, gmask)], np.array([p.pos for p in panos]))
    points = [x[~m] for x, m in zip(xs, gmask)]
    log(f"google base: {len(panos)} panos, {len(entries)} planes -> {n_surfaces} surfaces "
        f"({shared} shared by 2+ panos); ground {len(G)} points on one map  "
        f"[planes {t1 - t0:.1f}s, merge {t2 - t1:.1f}s, ground {time.time() - t2:.1f}s]")
    return Base(panos=panos, points=points, ground=G, ground_owner=owner)
