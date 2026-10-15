"""Buildings near DA3's points as points: roofs, walls and details."""
import numpy as np

from streetview_to_3d.build_scene.postprocess.world.Roof import Roof
from streetview_to_3d.build_scene.postprocess.world.buildings.Blocks import EDGE, INNER, ROOF, SURFACE, Blocks
from streetview_to_3d.build_scene.postprocess.world.buildings.tools.facades import enabled, facade_quads
from streetview_to_3d.build_scene.postprocess.world.tools.geometry import inset, sample_quads
from streetview_to_3d.build_scene.postprocess.world.tools.osm_tags import building_kind

SAMPLE = 16                   # reach judged from every SAMPLE-th point
REACH_M = 3.0                 # DA3 points this near make a building "reached"
INNER_M, INNER_GAP = (0.6, 1.5), 2.0   # inner walls this far in, this many times sparser, so gaps don't see through
ON_GROUND_M = 0.05            # wall points reach this far under the ground, no lower
CHUNK_M = 4.0                 # walls are spaced in pieces this long


def building_points(outlines, colours, land, site):
    """Blocks of the buildings (Outlines, coloured colours) with a DA3 point of site (Site) near
    their outline, standing on land (Land), spaced as the map's points are there; reached: the
    buildings DA3 reaches (the rest stand solid, solid_blocks)."""
    items = outlines.items
    may = np.flatnonzero(_reachable(items, site.points))
    blocks = _points([items[i] for i in may], site.spacing, land.surface, colours[may])
    blocks.which = may[blocks.which]
    blocks.reached = _reached(blocks, site.tree, len(items))
    return blocks


def _points(outlines, spacing, ground, colour):
    """Blocks of points for every outline: roof, walls (in CHUNK_M pieces), inner walls and details.

    spacing(xy): point spacing; ground(xy): final land height; colour: (n, 3)
    per building. A part starting in the air gets a flat underside."""
    roofs, walls, edge_of, us, vs, gaps, roof_gaps, edges = [], [], [], [], [], [], [], []
    roof_normals, wall_normals, kinds = [], [], []
    for xy, h, _, form, planes in outlines:
        foot = form.foot(xy, ground)
        base, top = foot + form.base_m, foot + h
        low = base - (0.0 if form.base_m else form.skirt_m)      # wall bottom
        roof = form.roof
        eaves = max(base, top - roof.height)
        # roof, spaced as at its middle; an underside if it starts in the air
        s = float(spacing(xy.mean(0)[None])[0])
        rp, rn = roof.surface(s)
        rp[:, 2] += eaves
        if form.base_m > 0:
            under = Roof(xy, "flat").surface(s)[0]
            rp = np.concatenate([rp, np.c_[under[:, :2], np.full(len(under), base)]])
            rn = np.concatenate([rn, np.tile([0.0, 0.0, -1.0], (len(under), 1))])
        roofs.append(rp[:, [0, 2, 1]] * [1, -1, 1])
        roof_normals.append(rn[:, [0, 2, 1]] * [1, -1, 1])   # east, down, north
        roof_gaps.append(np.full(len(rp), s))
        # walls: along each edge, up to the roof
        w, eo, uu, vv, gg, nn, kk = [], [], [], [], [], [], []
        for j, (a, c) in enumerate(zip(xy[:-1], xy[1:])):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            out = np.array([t[1], -t[0]])                    # outward for a counter-clockwise ring
            e = len(edges)
            edges.append((a, t, length, planes.get(j)))
            # pieces spaced as at their middle; ground under all columns at once
            pieces = []
            for c0 in np.arange(0, length, CHUNK_M):
                c1 = min(length, c0 + CHUNK_M)
                s = float(spacing((a + t * (c0 + c1) / 2)[None])[0])
                pieces.append((c0, s, np.arange(c0, c1, s)))
            columns = np.concatenate([p[2] for p in pieces])
            floors = ground(a + t * columns[:, None]) if not form.base_m else np.full(len(columns), low)
            for (c0, s, along), floor in zip(pieces, np.split(floors, np.cumsum([len(p[2]) for p in pieces])[:-1])):
                rise = roof.rise(a + t * along[:, None])
                # columns start at the ground (a part in the air: its base), rows aligned building-wide
                start = low - s * np.ceil(max(0.0, low - floor.min()) / s) if len(floor) else low
                levels = np.arange(start, eaves + (rise.max() if len(rise) else 0), s)
                U, V = np.repeat(along, len(levels)), np.tile(levels, len(along))
                ok = (V <= eaves + np.repeat(rise, len(levels)) + 1e-6) & \
                    (V >= np.repeat(floor, len(levels)) - ON_GROUND_M)
                U, V = U[ok], V[ok]
                kind = np.full(len(U), SURFACE)
                # corner and eaves points, as EDGE
                corner = np.arange(start, eaves + (rise[0] if len(rise) else 0), s) if c0 == 0 else np.zeros(0)
                corner = corner[corner >= (floor[0] if len(floor) else low) - ON_GROUND_M]
                top = eaves + rise
                U = np.r_[U, np.zeros(len(corner)), along]
                V = np.r_[V, corner, top]
                kind = np.r_[kind, np.full(len(corner) + len(along), EDGE)]
                w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                eo.append(np.full(len(U), e))
                uu.append(U)
                vv.append(V)
                gg.append(np.full(len(U), s))
                nn.append(np.tile([out[0], 0.0, out[1]], (len(U), 1)))
                kk.append(kind)
        # inner walls: further in, sparser
        for inset_m in INNER_M:
            ring = inset(xy, inset_m)
            for a, c in zip(ring[:-1], ring[1:]) if ring is not None else ():
                length = float(np.linalg.norm(c - a))
                if length < 1e-6:
                    continue
                t = (c - a) / length
                U, V, G = [], [], []
                for c0 in np.arange(0, length, CHUNK_M):
                    c1 = min(length, c0 + CHUNK_M)
                    s = INNER_GAP * float(spacing((a + t * (c0 + c1) / 2)[None])[0])
                    along, levels = np.arange(c0, c1, s), np.arange(low, eaves, s)
                    U.append(np.repeat(along, len(levels)))
                    V.append(np.tile(levels, len(along)))
                    G.append(np.full(len(along) * len(levels), s))
                U, V, G = np.concatenate(U), np.concatenate(V), np.concatenate(G)
                if not form.base_m and len(U):                            # from the ground up
                    keep = V >= ground(a + t * U[:, None]) - ON_GROUND_M
                    U, V, G = U[keep], V[keep], G[keep]
                w.append(np.column_stack([a[0] + t[0] * U, -V, a[1] + t[1] * U]))
                eo.append(np.full(len(U), INNER))
                uu.append(np.zeros(len(U)))
                vv.append(V)
                gg.append(G)
                nn.append(np.tile([t[1], 0.0, -t[0]], (len(U), 1)))
                kk.append(np.full(len(U), SURFACE))
        walls.append(np.concatenate(w) if w else np.zeros((0, 3)))
        edge_of.append(np.concatenate(eo) if eo else np.zeros(0, int))
        us.append(np.concatenate(uu) if uu else np.zeros(0))
        vs.append(np.concatenate(vv) if vv else np.zeros(0))
        gaps.append(np.concatenate(gg) if gg else np.zeros(0))
        wall_normals.append(np.concatenate(nn) if nn else np.zeros((0, 3)))
        kinds.append(np.concatenate(kk) if kk else np.zeros(0, int))
    if not roofs:
        z = np.zeros((0, 3))
        return Blocks(z, z, np.zeros(0, int), np.zeros(0, int), np.zeros(0), np.zeros(0), np.zeros(0), [],
                      np.zeros(0, bool), z, np.zeros(0, int))
    pts, cols, which, edge, u, v, gap, own = [], [], [], [], [], [], [], []
    normal, kind = [], []
    for i, (roof, wall, eo, uu, vv, rg, gg) in enumerate(zip(roofs, walls, edge_of, us, vs, roof_gaps, gaps)):
        normal += [roof_normals[i], wall_normals[i]]
        kind += [np.full(len(roof), SURFACE), kinds[i]]
        mine = outlines[i][3].roof_colour
        pts += [roof, wall]
        cols += [np.tile(np.clip(colour[i] if mine is None else mine, 0, 1), (len(roof), 1)),
                 np.tile(colour[i], (len(wall), 1))]
        which.append(np.full(len(roof) + len(wall), i))
        edge += [np.full(len(roof), ROOF), eo.astype(int)]
        u += [np.zeros(len(roof)), uu]
        v += [-roof[:, 1], vv]
        gap += [rg, gg]
        own += [np.full(len(roof), mine is not None), np.zeros(len(wall), bool)]
    for i, (xy, h, _, form, planes) in enumerate(outlines):
        step = float(max(spacing(xy).min(), .12))
        form.physical_facade = enabled(form, h, step)
        details = list(_detail_quads(xy, h, form, ground, colour[i], planes))
        if not details:
            continue
        quads = np.array([q for q, _ in details], float)
        facing = np.cross(quads[:, 1] - quads[:, 0], quads[:, 3] - quads[:, 0])
        size = np.linalg.norm(facing, axis=1)
        flat = size > 1e-9                                          # skip zero-area quads
        q, at = sample_quads(quads[flat], step)
        facing = (facing[flat] / size[flat, None])[at]
        normal.append(facing)
        kind.append(np.full(len(q), SURFACE))
        pts.append(q)
        cols.append(np.array([np.broadcast_to(c, 3) for _, c in details], float)[flat][at])
        which.append(np.full(len(q), i))
        edge.append(np.full(len(q), ROOF))
        u.append(np.zeros(len(q)))
        v.append(-q[:, 1])
        gap.append(np.full(len(q), step))
        own.append(np.ones(len(q), bool))
    return Blocks(np.concatenate(pts), np.concatenate(cols), np.concatenate(which), np.concatenate(edge),
                  np.concatenate(u), np.concatenate(v), np.concatenate(gap), edges, np.concatenate(own),
                  np.concatenate(normal), np.concatenate(kind))


def _detail_quads(xy, h, form, ground, colour, planes=None):
    """(quad (4, 3) world, colour) for physical facades (Form.physical_facade), roof trim and,
    without a facade, a simple entrance. Walls in planes (fitted to DA3) get none."""
    foot = form.foot(xy, ground)
    physical = form.physical_facade
    if physical:
        yield from facade_quads(xy, h, form, foot, colour, planes)
    eaves = foot + h - form.roof.height
    world = lambda en, up: np.c_[en[:, 0], -np.broadcast_to(up, len(en)), en[:, 1]]
    kind = building_kind(form.tags)
    if kind in ("roof", "carport", "hut"):
        return
    for j, (a, b) in enumerate(zip(xy[:-1], xy[1:])):
        length = np.linalg.norm(b - a)
        if planes and j in planes:
            continue
        if length < .5:
            continue
        t = (b - a) / length
        outward = np.array([t[1], -t[0]])
        if form.roof.shape == "flat":
            # parapet face and inward top cap
            z = eaves
            outer = np.array([a, b])
            inner = outer - outward * .22
            yield np.r_[world(outer, z - .4), world(outer[::-1], z)], np.clip(colour * 1.12, 0, 1)
            yield np.r_[world(outer, z), world(inner[::-1], z)], np.clip(colour * 1.2, 0, 1)
        else:
            z = eaves + form.roof.rise(np.array([a, b]))
            outer = np.array([a, b]) - outward * .02
            yield np.r_[world(outer, z - .18), world(outer[::-1], z[::-1])], colour * .75
        if not physical and j == form.front and length >= 3 and not form.base_m and not form.part \
                and kind not in ("shed", "garage", "garages", "warehouse", "industrial"):
            mid = (a + b) / 2
            ends = np.array([mid - t * .65, mid + t * .65]) + outward * .06
            top = min(foot + 2.35, eaves - .25)
            if top <= foot + 1.5:
                continue
            yield np.r_[world(ends, foot + .05), world(ends[::-1], top)], colour * .38 + [.02, .03, .04]
            canopy = ends + outward * .4
            yield np.r_[world(ends, top + .12), world(canopy[::-1], top + .12)], np.clip(colour * 1.15, 0, 1)


def _reachable(outlines, scene):
    """True for each outline with a scene point within REACH_M of its box (seen from above); only these get points."""
    out = np.zeros(len(outlines), bool)
    if not len(scene) or not outlines:
        return out
    xz = scene[:, [0, 2]]
    lo = xz.min(0)
    cell = np.floor((xz - lo) / REACH_M).astype(int)
    grid = np.zeros(cell.max(0) + 2, int)
    grid[cell[:, 0] + 1, cell[:, 1] + 1] = 1
    seen = grid.cumsum(0).cumsum(1)                     # occupied cells up to (i, j)
    top = np.array(seen.shape) - 2
    for k, (xy, *_) in enumerate(outlines):
        a = np.clip(np.floor((xy.min(0) - REACH_M - lo) / REACH_M).astype(int), 0, None)
        b = np.minimum(np.floor((xy.max(0) + REACH_M - lo) / REACH_M).astype(int), top)
        if (a <= b).all():
            out[k] = seen[b[0] + 1, b[1] + 1] - seen[a[0], b[1] + 1] - seen[b[0] + 1, a[1]] + seen[a[0], a[1]] > 0
    return out


def _reached(blocks, tree, n):
    """True for each of n buildings with a DA3 point (tree, or None) within REACH_M."""
    out = np.zeros(n, bool)
    if tree is not None and len(blocks.pts):
        d = tree.query(blocks.pts[::SAMPLE], distance_upper_bound=REACH_M, workers=-1)[0]
        out[np.unique(blocks.which[::SAMPLE][d < REACH_M])] = True
    return out
