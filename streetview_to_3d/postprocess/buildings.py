"""The buildings around the scene, from OpenStreetMap, stood on the terrain.

Footprints from osm.py, each raised into a block of points:

  - height: its own "height" tag, else building:levels x LEVEL_M, else a
    guess by kind (DEFAULT_M; most buildings carry neither -- NTU: 1864 of
    2628)
  - one DA3 built a wall of is slid onto it, a guessed height made DA3's
    (fit_to_scene); terrain.py then drops what DA3 already has of it and
    fades the rest in (seams.py)
  - standing on the lowest ground under its outline (terrain.py's map)
  - walls and a flat roof, spaced as the terrain is at that distance
  - roof coloured from the satellite straight above it; walls the
    building's own colour where the scene's panos see enough of it
    (pano_colours), else its roof colour darkened -- either shaded by
    which way the wall faces

Called by terrain.build, whose points they join in terrain.ply.
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


def points(outlines, step, ground, colour, sun):
    """(points (n, 3) world, colours (n, 3)) for every outline.

    step(d): spacing at d metres from the centre; ground(xy): height of the
    ground at east/north points; colour(xy): the satellite's RGB there."""
    roofs, walls, lights = [], [], []
    for xy, h, _ in outlines:
        s = step(float(np.linalg.norm(xy.mean(0))))
        base = ground(xy).min()
        top = base + h
        # roof: a grid over the outline, plus its corners
        lo, hi = xy.min(0), xy.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], s) + s / 2, np.arange(lo[1], hi[1], s) + s / 2)
        grid = np.stack([gx.ravel(), gy.ravel()], 1)
        roof = np.concatenate([grid[_inside(grid, xy)], xy[:-1]])
        roofs.append(np.column_stack([roof[:, 0], np.full(len(roof), -top), roof[:, 1]]))
        # walls: along each edge and up it; lit by how squarely it faces the sun
        levels = np.arange(base, top, s)
        w, light = [], []
        for a, c in zip(xy[:-1], xy[1:]):
            n = max(1, int(np.linalg.norm(c - a) / s))
            along = a + (c - a) * (np.arange(n) / n)[:, None]
            w.append(np.column_stack([np.repeat(along[:, 0], len(levels)), -np.tile(levels, n),
                                      np.repeat(along[:, 1], len(levels))]))
            edge = (c - a) / (np.linalg.norm(c - a) + 1e-9)
            light.append(np.full(n * len(levels), abs(np.array([edge[1], 0.0, -edge[0]]) @ sun)))
        walls.append(np.concatenate(w))
        lights.append(np.concatenate(light))
    if not roofs:
        return np.zeros((0, 3)), np.zeros((0, 3))
    rgb = colour(np.concatenate(roofs)[:, [0, 2]])
    pts, cols, which, lit, at = [], [], [], [], 0
    for i, (roof, wall, light) in enumerate(zip(roofs, walls, lights)):
        mine = rgb[at:at + len(roof)]
        at += len(roof)
        pts += [roof, wall]
        cols += [mine, mine.mean(0) * 0.85 * (0.7 + 0.3 * light[:, None])]
        which.append(np.full(len(roof) + len(wall), i))
        lit += [np.full(len(roof), np.nan), light]
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(which), np.concatenate(lit)


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
