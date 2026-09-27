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
5. never sideways: a fill point on a DA3 wall's line (any fill, merged or
   not, whose surface faces the same way) must lie between that wall's ends, measured from DA3's points on
   it; past every such wall's ends it is dropped. Height is free, so the
   fill still carries a wall above DA3's top.
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
ON_WALL_M, SIDE_M = 0.5, 0.2
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


def fill(base, da3, da3_normals, scene_points):
    """Google base points to add: the gaps of scene_points (DA3 and the one
    ground) filled from trusted panos, DA3's walls winning where both
    have one. da3/da3_normals: DA3's own points (for trust and walls)."""
    xs, ss, ks = [], [], []
    for k, (p, x, s) in enumerate(zip(base.panos, base.points, base.surfaces)):
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
    walls = []                                   # DA3 walls: (normal, d, along, ends)
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


def _onto_da3(n, d, x, cams, da3, da3_normals, walls):
    """x slid along each camera's line of sight onto DA3's copy of the wall
    (n, d), NaN rows where that is too far or DA3's faces too differently;
    None if DA3 has no copy. Records DA3's wall in walls."""
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
    if n2 @ n < np.cos(np.radians(AGREE_DEG)):
        return np.full_like(x, np.nan)
    d2 = n2 @ c
    along = np.cross(n2, [0.0, 1.0, 0.0])
    along /= np.linalg.norm(along)
    on = P[np.abs(P @ n2 - d2) < ON_WALL_M]
    if len(on) >= SAME_MIN:
        walls.append((n2, d2, along, np.percentile(on @ along, [1, 99])))
    ray = x - cams
    den = ray @ n2
    t = (d2 - cams @ n2) / np.where(np.abs(den) > 1e-6, den, np.nan)
    ok = np.isfinite(t) & (t > SLIDE[0]) & (t < SLIDE[1])
    return np.where(ok[:, None], cams + ray * np.nan_to_num(t)[:, None], np.nan)


def _within_walls(x, normals, walls):
    """False for fill on some DA3 wall's line (near its plane, its surface
    facing the same way) but past the ends of every such wall."""
    on_line = np.zeros(len(x), bool)
    inside = np.zeros(len(x), bool)
    for n2, d2, along, (a0, a1) in walls:
        line = (np.abs(x @ n2 - d2) < SAME_M) & (np.abs(normals @ n2) > np.cos(np.radians(SAME_DEG)))
        a = x @ along
        on_line |= line
        inside |= line & (a > a0 - SIDE_M) & (a < a1 + SIDE_M)
    return ~on_line | inside
