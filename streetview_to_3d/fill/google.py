"""Google's walls in DA3's gaps.

DA3 is the better shape, but it has holes: tall walls above where its views
reach, walls it dropped as unsure. The Google base (google_base) has those
walls as clean planes. Here they fill in, and where DA3 has the same wall,
DA3 wins:

1. trust: a Google pano whose depth map typically disagrees with DA3 by
   more than TRUST, looking the same way, is not used (one Stockholm pano
   put a wall 0.6 m away that is really 2-3 m)
2. gaps: from each trusted pano, a base point fills where DA3 has nothing
   along its line of sight within COVER_TOL of its distance -- grown by
   EXTEND_PX pixels so the fill reaches slightly into DA3's edge
3. walls only: surfaces more than WALL_DEG off vertical are dropped (the
   ground is one_ground's); each surface loses TRIM_M off its outer edge
   (blocky there) and surfaces under MIN_AREA_M2 are dropped as junk
4. DA3 wins: where DA3 has the same wall (points facing within SAME_DEG,
   within SAME_M of it, near its fill), a plane is fitted to them; if it
   faces within AGREE_DEG of Google's, the fill slides along the Google
   pano's lines of sight onto it (by SLIDE at most), else it is dropped.
   The overlap from 2 lands on DA3's wall, so the two join with no step.
5. never sideways: fill lying on a DA3 wall (within ON_WALL_M of its
   plane, facing the same way -- not a wall a metre off, nor the one
   across the street) is kept only along the stretches where DA3 has that
   wall. A stretch is found by sliding along the wall in SPAN_M steps over
   ALL of DA3's points on that plane: a step with SPAN_PER_M points per
   metre or more is wall, and BRIDGE_M or more of empty steps is a break
   (a doorway, the wall's end). Height is free, so the fill still carries
   a wall above DA3's top.
"""
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion
from scipy.spatial import cKDTree

TRUST, TRUST_MAX_M = 0.25, 25.0
COVER_TOL, EXTEND_PX, VIEW_W = 0.25, 3, 512
WALL_DEG = 10
TRIM_M, MIN_AREA_M2, FOOT_M = 0.5, 2.0, 0.25
SAME_DEG, SAME_M, SAME_NEAR_M, SAME_MIN = 40, 2.5, 5.0, 200
AGREE_DEG = 15
SLIDE = (0.5, 2.0)
ON_WALL_M, SPAN_M, BRIDGE_M, SPAN_PER_M = 0.5, 0.2, 0.6, 60
# Walls only from panos that need them (needs_walls): DA3's views reach ~29
# deg up, so a pano whose band just above that is mostly building is missing
# wall tops. Of its 12 directions (one per DA3 view), TALL_DIRECTIONS or more
# must be over TALL_SHARE building there. Stockholm's alley: 7 of 7 panos;
# NTU80, open with some tall blocks: 9 of 21 at 8, 14 at 6.
TALL_BAND_DEG, TALL_SHARE, TALL_DIRECTIONS = (30, 60), 0.3, 7
VOXEL_M = 0.05


def _look(p, x):
    """(u, v, distance) of world points x from Google pano p."""
    d = (x - p.pos) @ p.R.T
    r = np.linalg.norm(d, axis=1)
    return (np.arctan2(d[:, 0], d[:, 2]) / (2 * np.pi) + .5,
            np.arcsin(np.clip(d[:, 1] / np.maximum(r, 1e-9), -1, 1)) / np.pi + .5, r)


def _nearest(u, v, r, h, w):
    """Nearest distance per pixel of an h x w view, and each point's pixel."""
    px = np.clip((v * h).astype(int), 0, h - 1) * w + (u * w).astype(int) % w
    near = np.full(h * w, np.inf)
    np.minimum.at(near, px, r)
    return near, px


def trusted(p, da3):
    """Does Google pano p's own depth map agree with DA3 (points da3)?"""
    gh, gw = p.depth.shape
    near, _ = _nearest(*_look(p, da3), gh, gw)
    gd = p.depth.ravel()
    both = (gd > 0) & (near < TRUST_MAX_M)
    if both.sum() < 500:
        return True
    return np.median(np.abs(np.log(gd[both] / near[both]))) < np.log(1 + TRUST)


def needs_walls(labels):
    """Does a pano (its class map, segment.CLASSES ids) have building above
    DA3's reach in TALL_DIRECTIONS of its 12 directions?"""
    from streetview_to_3d.services.segment import LABEL_IDS
    h, w = labels.shape
    lo, hi = TALL_BAND_DEG
    band = np.isin(labels[int(h * (.5 - hi / 180)):int(h * (.5 - lo / 180))], [LABEL_IDS["building"], LABEL_IDS["wall"]])
    shares = [part.mean() for part in np.array_split(band, 12, axis=1)]
    return sum(s > TALL_SHARE for s in shares) >= TALL_DIRECTIONS


def fill(base, da3, da3_normals, scene_points, walls_from=None):
    """Google base points to add: the gaps of scene_points (DA3 and the one
    ground) filled from trusted panos, DA3's walls winning where both
    have one. da3/da3_normals: DA3's own points (for trust and walls).
    walls_from: the ids of the panos to take walls from (None: all)."""
    xs, ss, ks = [], [], []
    for k, (p, x, s) in enumerate(zip(base.panos, base.points, base.surfaces)):
        if walls_from is not None and p.id not in walls_from:
            continue
        if not len(x) or not trusted(p, da3):
            continue
        near, _ = _nearest(*_look(p, scene_points), VIEW_W // 2, VIEW_W)
        u, v, r = _look(p, x)
        px = np.clip((v * (VIEW_W // 2)).astype(int), 0, VIEW_W // 2 - 1) * VIEW_W + (u * VIEW_W).astype(int) % VIEW_W
        gap = np.zeros(near.size, bool)
        gap[px[near[px] > r * (1 + COVER_TOL)]] = True
        gap = binary_dilation(gap.reshape(VIEW_W // 2, VIEW_W), iterations=EXTEND_PX).ravel()
        keep = gap[px]
        xs.append(x[keep]); ss.append(s[keep]); ks.append(np.full(keep.sum(), k))
    if not xs:
        return np.zeros((0, 3))
    x, s, k = np.concatenate(xs), np.concatenate(ss), np.concatenate(ks)
    keep = _walls_only(base, x, s)
    cams = np.array([p.pos for p in base.panos])
    walls = []                                   # DA3 walls: (normal, d, along, stretches)
    for g in np.unique(s[keep]):
        m = keep & (s == g)
        slid = _onto_da3(base.planes[g][0], base.planes[g][1], x[m], cams[k[m]], da3, da3_normals, walls)
        if slid is None:
            continue
        idx = np.flatnonzero(m)
        keep[idx[~np.isfinite(slid[:, 0])]] = False
        ok = np.isfinite(slid[:, 0])
        x[idx[ok]] = slid[ok]
    normals = np.array([base.planes[g][0] for g in range(len(base.planes))])[s]
    keep &= _within_walls(x, normals, walls)
    x = x[keep]
    key = np.floor(x / VOXEL_M).astype(np.int64) + 2 ** 20
    _, first = np.unique((key[:, 0] << 42) | (key[:, 1] << 21) | key[:, 2], return_index=True)
    return x[first]


def _walls_only(base, x, s):
    """Vertical surfaces, trimmed at their own edges, big enough."""
    keep = np.zeros(len(x), bool)
    allx, alls = np.concatenate(base.points), np.concatenate(base.surfaces)
    er = int(round(TRIM_M / FOOT_M))
    for g in np.unique(s):
        n = base.planes[g][0]
        if abs(n[1]) > np.sin(np.radians(WALL_DEG)):
            continue
        e1 = np.cross(n, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        B = np.stack([e1, np.cross(n, e1)], 1)
        c_all = np.floor(allx[alls == g] @ B / FOOT_M).astype(int)
        lo = c_all.min(0) - er - 2
        foot = np.zeros(c_all.max(0) - lo + er + 3, bool)
        foot[tuple((c_all - lo).T)] = True
        foot = binary_erosion(binary_closing(foot, iterations=1), iterations=er)
        if foot.sum() * FOOT_M ** 2 < MIN_AREA_M2:
            continue
        m = s == g
        c = np.floor(x[m] @ B / FOOT_M).astype(int) - lo
        inside = np.all((c >= 0) & (c < foot.shape), axis=1)
        ok = np.zeros(m.sum(), bool)
        ok[inside] = foot[tuple(c[inside].T)]
        keep[np.flatnonzero(m)[ok]] = True
    return keep


def da3_copy(n, d, x, da3, da3_normals):
    """DA3's own copy of the wall plane (n, d) that the points x lie on:
    (n2, d2, agrees) fitted to DA3's points within SAME_M of it, facing
    within SAME_DEG of its way and within SAME_NEAR_M of x -- trees face
    every way and the ground up, so neither counts -- agrees when it faces
    within AGREE_DEG of n; None if DA3 has under SAME_MIN such points."""
    lo, hi = x.min(0) - SAME_NEAR_M, x.max(0) + SAME_NEAR_M
    cand = np.flatnonzero(np.all((da3 >= lo) & (da3 <= hi), axis=1))
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


def on_plane(n2, d2, da3, da3_normals):
    """True for DA3's points on the plane (n2, d2): within ON_WALL_M of it,
    facing its way."""
    return (np.abs(da3 @ n2 - d2) < ON_WALL_M) & (np.abs(da3_normals @ n2) > np.cos(np.radians(SAME_DEG)))


def _onto_da3(n, d, x, cams, da3, da3_normals, walls):
    """x slid along each camera's line of sight onto DA3's copy of the wall
    (n, d), NaN rows where that is too far or DA3's faces too differently;
    None if DA3 has no copy. Records DA3's wall in walls."""
    copy = da3_copy(n, d, x, da3, da3_normals)
    if copy is None:
        return None
    n2, d2, agrees = copy
    if not agrees:
        return np.full_like(x, np.nan)
    along = np.cross(n2, [0.0, 1.0, 0.0])
    along /= np.linalg.norm(along)
    # its length from ALL of DA3's points on that plane, not just those near
    # this piece of fill -- or the wall "ends" where the fill's neighbourhood does
    on = on_plane(n2, d2, da3, da3_normals)
    if on.sum() >= SAME_MIN:
        walls.append((n2, d2, along, _stretches(da3[on] @ along)))
    ray = x - cams
    den = ray @ n2
    t = (d2 - cams @ n2) / np.where(np.abs(den) > 1e-6, den, np.nan)
    ok = np.isfinite(t) & (t > SLIDE[0]) & (t < SLIDE[1])
    return np.where(ok[:, None], cams + ray * np.nan_to_num(t)[:, None], np.nan)


def _stretches(a):
    """[(start, end), ...] along a wall where DA3 has it: SPAN_M steps with
    at least SPAN_PER_M points per metre, gaps up to BRIDGE_M closed."""
    steps, count = np.unique(np.floor(a / SPAN_M).astype(int), return_counts=True)
    steps = steps[count >= SPAN_PER_M * SPAN_M]
    if not len(steps):
        return []
    breaks = np.flatnonzero((np.diff(steps) - 1) * SPAN_M >= BRIDGE_M)   # empty steps between
    starts = np.r_[steps[0], steps[breaks + 1]]
    ends = np.r_[steps[breaks], steps[-1]] + 1
    return [(a0 * SPAN_M, a1 * SPAN_M) for a0, a1 in zip(starts, ends)]


def _within_walls(x, normals, walls):
    """False for fill lying on some DA3 wall (within ON_WALL_M of its plane,
    its surface facing the same way) but along no stretch of any such wall."""
    on_line = np.zeros(len(x), bool)
    inside = np.zeros(len(x), bool)
    for n2, d2, along, stretches in walls:
        line = (np.abs(x @ n2 - d2) < ON_WALL_M) & (normals @ n2 > np.cos(np.radians(SAME_DEG)))
        a = x @ along
        on_line |= line
        for a0, a1 in stretches:
            inside |= line & (a >= a0) & (a <= a1)
    return ~on_line | inside
