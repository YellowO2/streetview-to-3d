"""Buildings DA3 never reaches as solid meshes."""
import numpy as np

from streetview_to_3d.postprocess.world.Roof import Roof
from streetview_to_3d.postprocess.world.buildings.tools.facades import facade_layout, glass
from streetview_to_3d.postprocess.world.tools.geometry import turning

SOLID_STEP_M = 2.0            # solid wall/roof triangles about this long
NO_FACADE = -1e4              # facade value for roof vertices


def solid_blocks(outlines, colours, blocks, land, site):
    """blocks.ply's mesh (write_mesh's arguments) of the buildings (Outlines, coloured colours) DA3
    does not reach (blocks.reached), standing on land (Land), each spaced as its nearest point
    of the map would be (site: Site); None if DA3 reaches them all."""
    solid = [o for o, r in zip(outlines.items, blocks.reached) if not r]
    if not solid:
        return None
    one = np.array([site.spacing(xy).min() for xy, *_ in solid])
    v, c, f, facade, g = _solid(solid, land.surface, colours[~blocks.reached], one)
    return dict(pts=v, cols=c, faces=f, facade=facade, gap=g)


def _wall_facade(form, h, u, v, glass_colour):
    """(n, 7) facade rows: metres along and up, bay, floor, window colour; NO_FACADE without windows."""
    layout = facade_layout(form, h)
    if layout is None:
        return np.full((len(u), 7), NO_FACADE)
    return np.c_[u, v, np.full(len(u), layout[0]), np.full(len(u), layout[1]), np.tile(glass_colour, (len(u), 1))]


def _solid(outlines, ground, colour, spacing=None):
    """Every outline as a mesh: (vertices, colours, faces, facade (n, 7)[, spacing]).

    Walls from ground(xy) (or the base for a part in the air) up to the roof,
    the roof's triangles, and a flat underside for parts in the air. facade
    rows as _wall_facade, NO_FACADE on roofs. spacing: one value per outline."""
    V, C, F, U, G = [], [], [], [], []
    count = 0

    def add(pts, col, faces, facade):
        nonlocal count
        V.append(pts)
        C.append(np.broadcast_to(col, pts.shape))
        F.append(faces + count)
        U.append(facade)
        G.append(np.full(len(pts), spacing[i] if spacing is not None else 0.0))
        count += len(pts)

    world = lambda en, up: np.c_[en[:, 0], -up, en[:, 1]]
    for i, (xy, h, _, form, *_) in enumerate(outlines):
        foot = form.foot(xy, ground)
        base, top = foot + form.base_m, foot + h
        low = base - (0.0 if form.base_m else form.skirt_m)      # wall bottom
        roof = form.roof
        eaves = max(base, top - roof.height)
        ring = xy if turning(xy) > 0 else xy[::-1]       # counter-clockwise: inside to the left
        for a, c in zip(ring[:-1], ring[1:]):
            length = float(np.linalg.norm(c - a))
            if length < 1e-6:
                continue
            t = (c - a) / length
            # corners where the roof bends and every SOLID_STEP_M, so the foot follows the ground
            u = np.unique(np.r_[length * roof.bends(a, c), np.arange(0, length, SOLID_STEP_M), length])
            en = a + t * u[:, None]
            up = eaves + roof.rise(en)
            if (up - base).max() < 1e-3:
                continue
            bottom = np.minimum(ground(en), up) if not form.base_m else np.full(len(u), low)
            pts = np.concatenate([world(en, bottom), world(en, up)])
            k = np.arange(len(u) - 1)
            n = len(u)
            faces = np.concatenate([np.c_[k, k + 1, n + k + 1], np.c_[k, n + k + 1, n + k]])
            add(pts, colour[i], faces, _wall_facade(form, h, np.r_[u, u], np.r_[bottom, up] - foot,
                                                   glass(colour[i])))
        tris = roof.triangles(SOLID_STEP_M)
        if form.base_m > 0:
            under = Roof(xy, "flat").triangles(SOLID_STEP_M)
            under[..., 2] = base - eaves
            tris = np.concatenate([tris, under[:, ::-1]])
        normal = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        keep = np.linalg.norm(normal, axis=1) > 1e-9
        tris, normal = tris[keep], normal[keep] / np.linalg.norm(normal[keep], axis=1, keepdims=True)
        if not len(tris):
            continue
        roof_colour = colour[i] if form.roof_colour is None else form.roof_colour
        pts = tris.reshape(-1, 3)
        add(world(pts[:, :2], eaves + pts[:, 2]), np.clip(roof_colour, 0, 1),
            np.arange(len(pts)).reshape(-1, 3), np.full((len(pts), 7), NO_FACADE))
    if not V:
        empty = np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int), np.zeros((0, 7))
        return empty if spacing is None else (*empty, np.zeros(0))
    # merge vertices identical in position, colour, facade and spacing
    rows = np.c_[np.concatenate(V), np.round(np.concatenate(C) * 255), np.concatenate(U),
                 np.concatenate(G)].astype(np.float32)
    rows, index = np.unique(rows, axis=0, return_inverse=True)
    out = rows[:, :3], rows[:, 3:6] / 255, index.ravel()[np.concatenate(F)], rows[:, 6:13]
    return out if spacing is None else (*out, rows[:, 13])
