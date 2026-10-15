"""Tunnel portals where a tunnel meets a drawn road."""
import numpy as np

from streetview_to_3d.postprocess.world.roads.Deck import SIDE
from streetview_to_3d.postprocess.world.tools.osm_roads import KEPT, kept_ways, width

PORTAL_H, PORTAL_SIDE_M = 5.0, 0.5         # tunnel opening height; extra width each side of the road
FRAME_M, LINTEL_M, MOUTH_M = 1.2, 1.5, 3.0  # frame pillar width, lintel height, dark interior depth
DARK = np.array([0.06, 0.06, 0.07])
PORTAL_MEET_M = 1.0   # a tunnel end this near a drawn road end gets a portal




def portals(network, height):
    """(points, colours, triangles) of a portal wherever one of network's (Network) tunnels
    meets a drawn road, at the road's height (RoadHeight) there."""
    return _portals(network.elements, network.to_xy, height)


def _portals(elements, to_xy, height):
    """(points, colours, triangles) of a portal wherever a kept tunnel meets a drawn road.

    Ends are matched by position (PORTAL_MEET_M; tile ways have no nodes). A
    concrete frame round a dark opening, at height(xy) of the road there."""
    from scipy.spatial import cKDTree
    ends = [w[1][k] for w in kept_ways(elements, to_xy) for k in (0, -1)]
    if not ends:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int)
    drawn_ends = cKDTree(np.array(ends))
    quads, colours = [], []
    for e in elements:
        tags = e.get("tags", {})
        if (e.get("type") != "way" or tags.get("highway") not in KEPT or tags.get("tunnel", "no") == "no"
                or len(e.get("geometry") or []) < 2):
            continue
        xy, w = to_xy(e["geometry"]), width(tags) / 2 + PORTAL_SIDE_M
        for p, q in ((xy[0], xy[1]), (xy[-1], xy[-2])):
            if drawn_ends.query(p)[0] > PORTAL_MEET_M:
                continue
            t = (q - p) / max(np.linalg.norm(q - p), 1e-9)          # into the tunnel
            s = np.array([-t[1], t[0]])
            h0 = float(height(p[None])[0])
            top, deep = h0 + PORTAL_H, p + t * MOUTH_M

            def wall(a, b, lo, hi, c):
                quads.append([(a, lo), (b, lo), (b, hi), (a, hi)])
                colours.append(c)
            wall(p - s * (w + FRAME_M), p - s * w, h0, top + LINTEL_M, SIDE)       # the frame
            wall(p + s * w, p + s * (w + FRAME_M), h0, top + LINTEL_M, SIDE)
            wall(p - s * w, p + s * w, top, top + LINTEL_M, SIDE)
            wall(deep - s * w, deep + s * w, h0, top, DARK)                      # the dark inside
            wall(p - s * w, deep - s * w, h0, top, DARK)
            wall(p + s * w, deep + s * w, h0, top, DARK)
            quads.append([(p - s * w, top), (p + s * w, top), (deep + s * w, top), (deep - s * w, top)])
            colours.append(DARK)
    if not quads:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros((0, 3), int)
    pts = np.array([[a[0], -h, a[1]] for quad in quads for a, h in quad])
    cols = np.repeat(np.array(colours), 4, axis=0)
    k = 4 * np.arange(len(quads))[:, None]
    faces = np.concatenate([k + [0, 1, 2], k + [0, 2, 3]])
    return pts, cols, faces
