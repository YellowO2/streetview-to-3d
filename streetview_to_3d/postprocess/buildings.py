"""The buildings around the scene, from OpenStreetMap, stood on the terrain.

Footprints from osm.py, each raised into a block of points:

  - height: its own "height" tag, else building:levels x LEVEL_M, else a
    guess by kind (DEFAULT_M; most buildings carry neither -- NTU: 1864 of
    2628)
  - one DA3 built a wall of is slid onto it, a guessed height made DA3's
    (fit_to_scene); what DA3 already has of it is left to DA3 and the rest
    faded in next to it, judged on the wall itself (seam)
  - standing on the lowest ground under its outline (terrain.py's map)
  - walls and a flat roof, spaced by how near the scene's cameras they
    are -- as densely as DA3 next to them (points)
  - roof coloured from the satellite straight above it; walls the
    building's own colour where the scene's panos see enough of it
    (pano_colours), else its roof colour darkened -- either shaded by
    which way the wall faces

Called by terrain.build, which writes them to buildings.ply.
"""
import numpy as np

LEVEL_M = 3.2
DEFAULT_M = {"house": 7, "detached": 7, "semidetached_house": 7, "terrace": 7, "bungalow": 5,
             "roof": 4, "shed": 3, "garage": 3, "garages": 3, "carport": 3, "hut": 3, "kiosk": 3}
DEFAULT_OTHER_M = 12.0
SEE_M = 150.0                 # panos colour the buildings this close to them
BEHIND_M, BEHIND = 3.0, 0.05  # how far behind what is in front a wall may stand and still be seen
SEEN_MIN = 20                 # pixels of a building the panos must see to colour it
FIT_M, FIT_MIN = 4.0, 200     # a building is DA3's where this many of its points stand this near its outline
SNAP_PASSES, SNAP_MAX_M = 8, 3.0
ON_WALL_M = 1.5               # after the snap, DA3's points this near an edge are on that wall
TOP_PERCENTILE = 97
MIN_HEIGHT_M = 2.5
SAMPLE_M = 0.25
CHUNK_M = 10.0                # a wall is spaced in pieces this long, each as at its middle
ROOF_MIN_STEP_M = 0.5         # roofs are seen from above only
COVER_M = 0.35                # an OSM point DA3 has a point this near (on the wall) is DA3's
SEAM_DEPTH_M, SEAM_FADE_M = 3.0, 3.0   # DA3's copy of a wall: this far in front or behind; the seam's width
SEAM_TINT, SEAM_MIN = 0.8, 20


def _height(tags):
    """(height m, whether it is only a guess)."""
    for key, per in (("height", 1.0), ("building:levels", LEVEL_M)):
        try:
            v = float(str(tags[key]).split()[0].replace(",", "."))
            if v > 0:
                return v * per, False
        except (KeyError, ValueError):
            pass
    return DEFAULT_M.get(tags.get("building"), DEFAULT_OTHER_M), True


def outlines(elements, to_xy):
    """[(outline (n, 2) east/north metres, closed, height m, guessed)] of
    the buildings among osm.fetch's elements; to_xy(lat, lon) -> east,
    north."""
    out = []
    for e in elements:
        if "building" not in e.get("tags", {}):
            continue
        rings = [e.get("geometry")] if e["type"] == "way" else \
            [m.get("geometry") for m in e.get("members", []) if m.get("role") == "outer"]
        for ring in rings:
            if not ring or len(ring) < 4 or ring[0] != ring[-1]:
                continue                  # a relation's outer split over several ways: left out
            out.append((to_xy(ring), *_height(e["tags"])))
    return out


def _edge_samples(xy, step=SAMPLE_M):
    """Points every step metres along a closed outline's edges."""
    parts = []
    for a, c in zip(xy[:-1], xy[1:]):
        n = max(1, int(np.linalg.norm(c - a) / step))
        parts.append(a + (c - a) * (np.arange(n) / n)[:, None])
    return np.concatenate(parts)


def fit_to_scene(outlines, walls, ground):
    """outlines, each building DA3 built a wall of moved onto it and, where
    its height was a guess, given DA3's.

    walls: the scene's points standing above the ground (world, (n, 3)).
    A building counts as DA3's where FIT_MIN of them stand within FIT_M of
    its outline; it is then slid (never turned: OSM's shape is good, its
    place a metre or two off) by the robust median of those points' offsets
    to its nearest edge, SNAP_PASSES times (not at all if that runs past
    SNAP_MAX_M: then something else is pulling it, NTU's 3 m shelter roof
    over trees) -- and a
    guessed height becomes the top of DA3's points on its walls
    (TOP_PERCENTILE), DA3 having seen that far at least. Returns the new
    outlines and how many moved."""
    from scipy.spatial import cKDTree
    if not len(walls):
        return outlines, 0
    tree = cKDTree(walls[:, [0, 2]])
    out, moved = [], 0
    for xy, h, guessed in outlines:
        edge = _edge_samples(xy)
        near = tree.query_ball_point(edge, FIT_M)
        idx = np.unique(np.concatenate([np.asarray(i, int) for i in near])) if len(near) else []
        if len(idx) < FIT_MIN:
            out.append((xy, h, guessed))
            continue
        p = walls[idx]
        shift = np.zeros(2)
        for _ in range(SNAP_PASSES):
            d, k = cKDTree(edge + shift).query(p[:, [0, 2]])
            close = d < FIT_M
            shift += np.median(p[close][:, [0, 2]] - (edge + shift)[k[close]], 0)
            if np.linalg.norm(shift) > SNAP_MAX_M:
                break
        if np.linalg.norm(shift) > SNAP_MAX_M:
            shift = np.zeros(2)           # pulled away by something else (trees under a shelter's roof): left where OSM has it
        xy = xy + shift
        if guessed:
            on = cKDTree(edge + shift).query(p[:, [0, 2]])[0] < ON_WALL_M
            if on.sum() >= FIT_MIN:
                top = np.percentile(-p[on, 1], TOP_PERCENTILE)
                h = max(MIN_HEIGHT_M, top - ground(xy).min())
        out.append((xy, h, guessed))
        moved += bool(shift.any())
    return out, moved


def _inside(pts, ring):
    """True where pts (m, 2) fall inside the closed ring (n, 2)."""
    a, b = ring[:-1], ring[1:]
    x, y = pts[:, :1], pts[:, 1:]
    crosses = ((a[:, 1] > y) != (b[:, 1] > y)) & \
        (x < (b[:, 0] - a[:, 0]) * (y - a[:, 1]) / (b[:, 1] - a[:, 1] + 1e-12) + a[:, 0])
    return crosses.sum(1) % 2 == 1


class Blocks:
    """Every building's points, and for each where on its building it is:
    which building, how lit (NaN on a roof), which wall edge (-1 on a
    roof) and its place on that wall, u metres along and v metres up
    (height above sea level). edges[e]: (start, along, outward, length)."""

    def __init__(self, pts, cols, which, light, edge, u, v, edges):
        self.pts, self.cols, self.which, self.light = pts, cols, which, light
        self.edge, self.u, self.v, self.edges = edge, u, v, edges

    def take(self, keep):
        for k in ("pts", "cols", "which", "light", "edge", "u", "v"):
            setattr(self, k, getattr(self, k)[keep])


def points(outlines, spacing, ground, colour, sun):
    """Blocks for every outline.

    spacing(xy): point spacing at east/north points -- a wall is laid in
    pieces up to CHUNK_M long, each spaced as at its middle, so only what
    is near gets dense; ground(xy): the ground's height there; colour(xy):
    the satellite's RGB there."""
    roofs, walls, lights, edge_of, us, vs, edges = [], [], [], [], [], [], []
    for xy, h, _ in outlines:
        base = ground(xy).min()
        top = base + h
        # roof: a grid over the outline, plus its corners, spaced as at its middle
        s = float(max(spacing(xy.mean(0)[None])[0], ROOF_MIN_STEP_M))
        lo, hi = xy.min(0), xy.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], s) + s / 2, np.arange(lo[1], hi[1], s) + s / 2)
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        roof = np.concatenate([grid[_inside(grid, xy)], xy[:-1]])
        roofs.append(np.column_stack([roof[:, 0], np.full(len(roof), -top), roof[:, 1]]))
        # walls: along each edge and up it; lit by how squarely it faces the sun
        w, light, eo, uu, vv = [], [], [], [], []
        for a, c in zip(xy[:-1], xy[1:]):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            out = np.array([t[1], -t[0]])                    # outward for a counter-clockwise ring
            e = len(edges)
            edges.append((a, t, out, length))
            lit = abs(np.array([out[0], 0.0, out[1]]) @ sun)
            for c0 in np.arange(0, length, CHUNK_M):
                c1 = min(length, c0 + CHUNK_M)
                s = float(spacing((a + t * (c0 + c1) / 2)[None])[0])
                along = np.arange(c0, c1, s)
                levels = np.arange(base, top, s)
                U, V = np.repeat(along, len(levels)), np.tile(levels, len(along))
                w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                light.append(np.full(len(U), lit))
                eo.append(np.full(len(U), e))
                uu.append(U)
                vv.append(V)
        walls.append(np.concatenate(w) if w else np.zeros((0, 3)))
        lights.append(np.concatenate(light) if light else np.zeros(0))
        edge_of.append(np.concatenate(eo) if eo else np.zeros(0, int))
        us.append(np.concatenate(uu) if uu else np.zeros(0))
        vs.append(np.concatenate(vv) if vv else np.zeros(0))
    if not roofs:
        z = np.zeros((0, 3))
        return Blocks(z, z, np.zeros(0, int), np.zeros(0), np.zeros(0, int), np.zeros(0), np.zeros(0), [])
    rgb = colour(np.concatenate(roofs)[:, [0, 2]])
    pts, cols, which, lit, edge, u, v, at = [], [], [], [], [], [], [], 0
    for i, (roof, wall, light, eo, uu, vv) in enumerate(zip(roofs, walls, lights, edge_of, us, vs)):
        mine = rgb[at:at + len(roof)]
        at += len(roof)
        pts += [roof, wall]
        cols += [mine, mine.mean(0) * 0.85 * (0.7 + 0.3 * light[:, None])]
        which.append(np.full(len(roof) + len(wall), i))
        lit += [np.full(len(roof), np.nan), light]
        edge += [np.full(len(roof), -1), eo.astype(int)]
        u += [np.zeros(len(roof)), uu]
        v += [-roof[:, 1], vv]
    return Blocks(np.concatenate(pts), np.concatenate(cols), np.concatenate(which), np.concatenate(lit),
                  np.concatenate(edge), np.concatenate(u), np.concatenate(v), edges)


def seam(blocks, walls, wall_cols, roofs_near, rng):
    """Leave to DA3 what it has of each building, and fade the rest in.

    A wall's points against DA3's own copy of that wall, on the wall: DA3's
    points within SEAM_DEPTH_M in front of or behind it (walls: the scene's
    points above its ground), flattened onto it as (along, up). An OSM
    point DA3 covers (within COVER_M there) goes, however far in front or
    behind DA3's copy stands -- a 3D distance left a strip as wide as that
    gap empty. Next to what DA3 covers, over SEAM_FADE_M: half of them
    kept rising to all, pulled onto DA3's depth for that wall (its median)
    and to DA3's colour there -- the two meet and interleave rather than
    stop. A roof goes where DA3 has points within COVER_M of it
    (roofs_near: their distance).
    Returns how many points DA3 already had."""
    from scipy.spatial import cKDTree
    from streetview_to_3d.postprocess.seams import ramp
    keep = np.ones(len(blocks.pts), bool)
    keep[blocks.edge < 0] = roofs_near[blocks.edge < 0] > COVER_M
    if not len(walls):
        blocks.take(keep)
        return int((~keep).sum())
    tree = cKDTree(walls[:, [0, 2]])
    order = np.argsort(blocks.edge, kind="stable")
    bounds = np.searchsorted(blocks.edge[order], np.arange(len(blocks.edges) + 1))
    for e, (a, t, out, length) in enumerate(blocks.edges):
        mine = order[bounds[e]:bounds[e + 1]]
        if not len(mine):
            continue
        near = tree.query_ball_point(a + t * length / 2, length / 2 + SEAM_DEPTH_M)
        if len(near) < SEAM_MIN:
            continue
        near = np.asarray(near)
        rel = walls[near][:, [0, 2]] - a
        u, n = rel @ t, rel @ out
        on = (np.abs(n) < SEAM_DEPTH_M) & (u > -SEAM_FADE_M) & (u < length + SEAM_FADE_M)
        if on.sum() < SEAM_MIN:
            continue
        near, u, n = near[on], u[on], n[on]
        d, k = cKDTree(np.c_[u, -walls[near, 1]]).query(np.c_[blocks.u[mine], blocks.v[mine]])
        band = ramp(d / SEAM_FADE_M)
        keep[mine] = (d > COVER_M) & (rng.random(len(mine)) < 0.5 + 0.5 * band)
        pull = (1 - band) * np.median(n)
        blocks.pts[mine, 0] += pull * out[0]
        blocks.pts[mine, 2] += pull * out[1]
        w = (SEAM_TINT * (1 - band))[:, None]
        blocks.cols[mine] = blocks.cols[mine] * (1 - w) + wall_cols[near[k]] * w
    blocks.take(keep)
    return int(len(keep) - keep.sum())


def pano_colours(pts, which, n, cameras, photos, occluders):
    """(n, 3) each building's colour as the panos see it, NaN where too
    little of it is seen.

    A wall point is seen by a camera within SEE_M when nothing is more than
    BEHIND_M (+ BEHIND of the distance) in front of it -- DA3's own copy of
    the wall stands a metre or two off OSM's outline and must not hide it,
    a tree or a nearer building further off does -- and its pixel is
    labelled building or wall (never sky over a building guessed too tall,
    nor a car). A building's colour is the median over every such pixel,
    SEEN_MIN of them at least: one flat colour, robust to a stray one."""
    from streetview_to_3d.fill.paint import NADIR_DEG, ZB_W, _at
    from streetview_to_3d.services.segment import LABEL_IDS
    ids = [LABEL_IDS["building"], LABEL_IDS["wall"]]
    everything = np.concatenate([occluders, pts])
    samples, owners = [], []
    h, w = ZB_W // 2, ZB_W
    for cam, ph in zip(cameras, photos):
        if ph is None:
            continue
        u, v, r, _ = cam.look(everything)
        near = np.full(h * w, np.inf)
        o = np.argsort(-r)
        iu, iv = (u[o] * w).astype(int), (v[o] * h).astype(int)
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                near[np.clip(iv + dv, 0, h - 1) * w + (iu + du) % w] = r[o]
        u, v, r, below = cam.look(pts)
        px = np.clip((v * h).astype(int), 0, h - 1) * w + (u * w).astype(int) % w
        seen = (r < SEE_M) & (below < NADIR_DEG) & (r <= near[px] + BEHIND_M + BEHIND * r)
        seen &= np.isin(_at(ph[1], u, v), ids) & ~_at(ph[2], u, v)
        if seen.any():
            from PIL import Image
            img = np.asarray(Image.open(ph[0]).convert("RGB"))
            samples.append(_at(img, u[seen], v[seen]) / 255.0)
            owners.append(which[seen])
    out = np.full((n, 3), np.nan)
    if not samples:
        return out
    samples, owners = np.concatenate(samples), np.concatenate(owners)
    order = np.argsort(owners, kind="stable")
    samples, owners = samples[order], owners[order]
    starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(owners)]):
        if b - a >= SEEN_MIN:
            out[owners[a]] = np.median(samples[a:b], 0)
    return out
