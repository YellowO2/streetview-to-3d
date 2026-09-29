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
FIT_M = 5.0                   # buildings this far past the scene's points are not looked at
WALL_SAMPLE_M = 1.0
SNAP_MAX_M = 2.0              # OSM's real offsets were 0.2-1.3 m (Stockholm, NTU)
ROAD_SLACK = 0.02             # a slide may put this much more of an outline onto roads
PULL_MAX_M = 1.0              # a wall's seam pulled onto DA3's plane by at most this
TOP_PERCENTILE = 97
MIN_HEIGHT_M = 2.5
CHUNK_M = 4.0                 # a wall is spaced in pieces this long, each as at its middle
ROOF_MIN_STEP_M = 0.5         # roofs are seen from above only
COVER_M = 0.35                # an OSM point DA3 has a point this near (on the wall) is DA3's
SEAM_FADE_M = 3.0             # the seam's width
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


def _wall_samples(a, c, base, top):
    """Points every WALL_SAMPLE_M over the wall from a to c, base to top."""
    length = float(np.linalg.norm(c - a))
    u = np.arange(0, length + 1e-9, WALL_SAMPLE_M)
    v = np.arange(base, top + 1e-9, WALL_SAMPLE_M)
    xy = a + (c - a) * (u / max(length, 1e-9))[:, None]
    return np.column_stack([np.repeat(xy[:, 0], len(v)), -np.tile(v, len(u)), np.repeat(xy[:, 1], len(v))])


def fit_to_scene(outlines, da3, da3_normals, ground, on_road=None):
    """outlines, each building DA3 built a wall of moved onto it and, where
    its height was a guess, given DA3's; each outline gains {edge: (n2,
    d2)}, DA3's plane for each of its walls DA3 has.

    Each wall looks for DA3's copy of it as the fill does for Google's
    (fill.google.da3_copy: DA3's points near its plane that face its way,
    so trees, poles and the ground never count). A copy that agrees says
    how far that wall is off, along its own facing; one slide of the whole
    building (never turned: OSM's shape is good, its place a metre or two
    off) meets all of them in least squares -- with one wall, or parallel
    ones, it only moves towards them, never sideways along them. Not moved
    if that is over SNAP_MAX_M, or if it moves the outline further onto a
    road (on_road(xy): the share of it there) -- buildings do not stand in
    roads. A guessed height becomes the top of DA3's points on its walls
    (TOP_PERCENTILE). Returns the new outlines and how many moved."""
    from streetview_to_3d.fill.google import da3_copy, on_plane
    if not len(da3):
        return [o + ({},) for o in outlines], 0
    lo, hi = da3[:, [0, 2]].min(0) - FIT_M, da3[:, [0, 2]].max(0) + FIT_M
    out, moved = [], 0
    for xy, h, guessed in outlines:
        if not ((xy.max(0) >= lo) & (xy.min(0) <= hi)).all():
            out.append((xy, h, guessed, {}))
            continue
        base = ground(xy).min()
        planes, rows, want, weight, tops = {}, [], [], [], []
        for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            n = np.array([t[1], 0.0, -t[0]])
            copy = da3_copy(n, float(n @ [a[0], 0, a[1]]), _wall_samples(a, c, base, base + h), da3, da3_normals)
            if copy is None or not copy[2]:
                continue
            n2, d2, _ = copy
            on = on_plane(n2, d2, da3, da3_normals)
            planes[j] = (n2, d2)
            rows.append(n2[[0, 2]])
            want.append(d2 - n2 @ [a[0], 0, a[1]])
            weight.append(np.sqrt(on.sum()))
            tops.append(-da3[on, 1])
        if not planes:
            out.append((xy, h, guessed, {}))
            continue
        w = np.array(weight)[:, None]
        shift = np.linalg.lstsq(np.array(rows) * w, np.array(want) * w[:, 0], rcond=None)[0]
        if np.linalg.norm(shift) > SNAP_MAX_M or (on_road and on_road(xy + shift) > on_road(xy) + ROAD_SLACK):
            shift = np.zeros(2)
        xy = xy + shift
        if guessed:
            top = np.percentile(np.concatenate(tops), TOP_PERCENTILE)
            h = max(MIN_HEIGHT_M, top - base)
        out.append((xy, h, guessed, planes))
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
    (height above sea level). edges[e]: (start, along, outward, length,
    DA3's plane for it (n2, d2) or None)."""

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
    for xy, h, _, planes in outlines:
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
        for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            out = np.array([t[1], -t[0]])                    # outward for a counter-clockwise ring
            e = len(edges)
            edges.append((a, t, out, length, planes.get(j)))
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


def seam(blocks, da3, da3_normals, da3_cols, roofs_near, rng):
    """Leave to DA3 what it has of each building, and fade the rest in.

    Only walls DA3 has a copy of (fit_to_scene's planes) are touched, and
    only against DA3's points on that plane (fill.google.on_plane), seen
    on the wall as (along, up): an OSM point DA3 covers there (within
    COVER_M) goes, however far in front or behind DA3's copy stands. Next
    to what DA3 covers, over SEAM_FADE_M: half of them kept rising to all,
    pulled onto DA3's plane (by at most PULL_MAX_M) and to its colour
    there -- the two meet and interleave rather than stop. A roof goes
    where DA3 has points within COVER_M of it (roofs_near: their
    distance). Returns how many points DA3 already had."""
    from scipy.spatial import cKDTree
    from streetview_to_3d.fill.google import on_plane
    from streetview_to_3d.postprocess.seams import ramp
    keep = np.ones(len(blocks.pts), bool)
    keep[blocks.edge < 0] = roofs_near[blocks.edge < 0] > COVER_M
    order = np.argsort(blocks.edge, kind="stable")
    bounds = np.searchsorted(blocks.edge[order], np.arange(len(blocks.edges) + 1))
    for e, (a, t, out, length, plane) in enumerate(blocks.edges):
        mine = order[bounds[e]:bounds[e + 1]]
        if plane is None or not len(mine):
            continue
        n2, d2 = plane
        box = np.all(np.abs(da3[:, [0, 2]] - (a + t * length / 2)) < length / 2 + SEAM_FADE_M + 1, axis=1)
        idx = np.flatnonzero(box)
        idx = idx[on_plane(n2, d2, da3[idx], da3_normals[idx])]
        if len(idx) < SEAM_MIN:
            continue
        u = (da3[idx][:, [0, 2]] - a) @ t
        d, k = cKDTree(np.c_[u, -da3[idx, 1]]).query(np.c_[blocks.u[mine], blocks.v[mine]])
        band = ramp(d / SEAM_FADE_M)
        keep[mine] = (d > COVER_M) & (rng.random(len(mine)) < 0.5 + 0.5 * band)
        off = np.clip(d2 - blocks.pts[mine] @ n2, -PULL_MAX_M, PULL_MAX_M)
        blocks.pts[mine] += ((1 - band) * off)[:, None] * n2
        w = (SEAM_TINT * (1 - band))[:, None]
        blocks.cols[mine] = blocks.cols[mine] * (1 - w) + da3_cols[idx[k]] * w
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
