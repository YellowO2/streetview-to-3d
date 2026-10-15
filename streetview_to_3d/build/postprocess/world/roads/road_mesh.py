"""The roads' surface away from the cameras, as triangles."""
import numpy as np

from streetview_to_3d.build.postprocess.world.roads.Deck import DECK_M, SIDE
from streetview_to_3d.build.postprocess.world.roads.tools.paving import (
    KERB, KERB_M, LIFT_M, PAVEMENT, PAVING_JOINT, STRIP_M, curb_faces, surface_colours,
)
from streetview_to_3d.build.postprocess.world.roads.tools.street_details import street_details
from streetview_to_3d.build.postprocess.world.tools.reach import NEAR_M, RADIUS_M

BELOW_M = 0.05        # the mesh sits this far under the points so the points show


def road_mesh(network, height, site, mouths):
    """roads.ply's mesh (write_mesh's arguments): network's surface (Network) and its bridges
    (height.decks) past NEAR_M of site's cameras (Site), at height (RoadHeight), and the tunnel
    mouths; spaced as the map's points are there (Site.spacing)."""
    import shapely
    mesh = None
    if network.shapes:
        near_cams = shapely.union_all(shapely.buffer(shapely.points(site.cameras), NEAR_M, quad_segs=16))
        mesh = _surface(network, height,
                        shapely.difference(shapely.box(-RADIUS_M, -RADIUS_M, RADIUS_M, RADIUS_M), near_cams))
    far = [b for b in height.decks if not site.from_cameras(b.xy).min() < NEAR_M]
    if height.decks:
        mesh = _merged(mesh, _bridge_surface(far))
    pts, cols, faces = _merged(mesh, mouths)
    return dict(pts=pts, cols=cols, faces=faces, gap=site.spacing(pts[:, [0, 2]]))


def _merged(a, b):
    """Two (points, colours, triangles) meshes as one; a may be None."""
    if a is None or not len(a[2]):
        return b
    if not len(b[2]):
        return a
    return np.concatenate([a[0], b[0]]), np.concatenate([a[1], b[1]]), np.concatenate([a[2], b[2] + len(a[0])])


def _surface(net, ground, away):
    """(points, colours, triangles) of the Network's surface LIFT_M over ground(xy), within away."""
    import shapely
    pts, cols, faces, n = [], [], [], 0
    for colour, shape in net.shapes.items():
        area = shapely.segmentize(shapely.intersection(shape, away), STRIP_M)
        tris = shapely.get_parts(shapely.constrained_delaunay_triangles(area)) if not area.is_empty else []
        if not len(tris):
            continue
        corners = shapely.get_coordinates(tris).reshape(-1, 4, 2)[:, :3]
        xy, idx = np.unique(np.round(corners.reshape(-1, 2), 3), axis=0, return_inverse=True)
        pts.append(np.column_stack([xy[:, 0], -(ground(xy) + LIFT_M - BELOW_M + (KERB_M if colour in (PAVEMENT, KERB, PAVING_JOINT) else 0)), xy[:, 1]]))
        cols.append(surface_colours(xy, colour))
        faces.append(idx.reshape(-1, 3) + n)
        n += len(xy)
    for quad in curb_faces(net, ground, away):
        pts.append(quad)
        cols.append(np.tile(np.array(KERB) * .82, (4, 1)))
        faces.append(np.array([[0, 1, 2], [0, 2, 3]]) + n)
        n += 4
    for quad, colour in street_details(net, ground):
        if not shapely.intersects(shapely.Polygon(quad[:, [0, 2]]).convex_hull, away):
            continue
        pts.append(quad)
        cols.append(np.tile(colour, (4, 1)))
        faces.append(np.array([[0, 1, 2], [0, 2, 3]]) + n)
        n += 4
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(faces)


def _bridge_surface(decks_):
    """(points, colours, triangles) of each deck and its DECK_M edge down each side."""
    pts, cols, faces, n = [], [], [], 0
    for deck in decks_:
        centre, h, side = deck.sample(STRIP_M)
        width, colour = deck.width, deck.colour
        k = len(centre)
        rows = []            # corner lines: deck left, right; then each side's top and bottom
        for edge in (-width / 2, width / 2):
            rows.append((centre + edge * side, h, colour))
        for edge in (-width / 2, width / 2):
            rows += [(centre + edge * side, h, SIDE), (centre + edge * side, h - DECK_M, SIDE)]
        for xy_, h_, c in rows:
            pts.append(np.column_stack([xy_[:, 0], -h_, xy_[:, 1]]))
            cols.append(np.tile(c, (k, 1)))
        i = np.arange(k - 1)
        for a in (0, 2, 4):               # the deck, then each side: a strip between rows a and a + 1
            A, B = n + a * k + i, n + (a + 1) * k + i
            faces += [np.stack([A, B, B + 1], 1), np.stack([A, B + 1, A + 1], 1)]
        n += 6 * k
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int)
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(faces)
