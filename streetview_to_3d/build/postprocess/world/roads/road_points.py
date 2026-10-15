"""The roads near the cameras as points, coloured from the panos."""
import numpy as np

from streetview_to_3d.build.postprocess.world.roads.Deck import DECK_M, SIDE
from streetview_to_3d.build.postprocess.world.roads.RoadPoints import RoadPoints
from streetview_to_3d.build.postprocess.world.roads.tools.paving import (
    KERB, KERB_M, LIFT_M, PAVEMENT, PAVING_JOINT, curb_faces, surface_colours,
)
from streetview_to_3d.build.postprocess.world.roads.tools.street_details import street_details
from streetview_to_3d.build.postprocess.world.tools.geometry import sample_quads
from streetview_to_3d.build.postprocess.world.tools.pano_paint import pano_paint
from streetview_to_3d.build.postprocess.world.tools.reach import NEAR_M
from streetview_to_3d.build.postprocess.world.tools.seams import toward


def road_points(network, height, site, water, views):
    """RoadPoints of network's surface (Network) and its bridges (height.decks) within NEAR_M of
    site's cameras (Site), at height (RoadHeight): none where the scene has ground of its own or
    on the water (Water), nor on a bridge where the scene has its own; painted from the panos
    (views: PanoViews) near the cameras and turning into DA3's points near them."""
    near = lambda xy: site.from_cameras(xy) < NEAR_M
    pts, cols = np.zeros((0, 3)), np.zeros((0, 3))
    if network.shapes:
        rp, rc = _points(network, site.spacing, height, near)
        keep = ~site.ground.covers(rp[:, [0, 2]])
        keep &= ~water.inside(rp[:, [0, 2]])
        rp = rp[keep]
        pts, cols = np.concatenate([pts, rp]), np.concatenate([cols, rc[keep]])
    close = [b for b in height.decks if site.from_cameras(b.xy).min() < NEAR_M]
    if close:
        bp_, bc_ = _bridge_points(close, site.spacing)
        d, g, _ = site.ground.at(bp_[:, [0, 2]])
        keep = d > 0                                     # the scene's own bridge wins
        bp_, bc_ = bp_[keep], bc_[keep]
        pts, cols = np.concatenate([pts, bp_]), np.concatenate([cols, bc_])
    cols, painted = pano_paint(views, pts, cols, site.points)
    cols, _ = toward(pts, cols, site.tree, site.colours)
    return RoadPoints(pts, cols, site.spacing(pts[:, [0, 2]]), painted)


def _points(net, step, ground, where):
    """(points, colours) of the Network's surface LIFT_M over ground(xy), where(xy) only.

    Spaced step(xy) apart; each point kept only inside its colour's shape."""
    import shapely
    xys, cols = [], []
    for xy, width, colour in net.strips:
        for a, b in zip(xy[:-1], xy[1:]):
            length = np.linalg.norm(b - a)
            if length < 1e-6:
                continue
            s = float(step(((a + b) / 2)[None])[0])
            if s > width:
                continue
            along = a + (b - a) * (np.arange(max(1, int(length / s)) + 1) / max(1, int(length / s)))[:, None]
            side = np.array([-(b - a)[1], (b - a)[0]]) / length
            across = np.arange(-width / 2, width / 2 + 1e-9, s)
            xys.append((along[:, None, :] + across[None, :, None] * side).reshape(-1, 2))
            cols.append(np.tile(colour, (len(xys[-1]), 1)))
    for xy, colour in net.fills:
        s = float(step(xy.mean(0)[None])[0])
        lo, hi = xy.min(0), xy.max(0)
        counts = np.maximum(1, np.ceil((hi - lo) / s).astype(int))
        gx, gy = np.meshgrid(lo[0] + (np.arange(counts[0]) + .5) * (hi[0] - lo[0]) / counts[0],
                             lo[1] + (np.arange(counts[1]) + .5) * (hi[1] - lo[1]) / counts[1])
        xys.append(np.stack([gx.ravel(), gy.ravel()], 1))
        cols.append(np.tile(colour, (len(xys[-1]), 1)))
    if not xys:
        return np.zeros((0, 3)), np.zeros((0, 3))
    xy, col = np.concatenate(xys), np.concatenate(cols)
    keep = where(xy)
    xy, col = xy[keep], col[keep]
    mine = np.zeros(len(xy), bool)
    for colour, shape in net.shapes.items():
        of = np.flatnonzero((col == colour).all(1))
        if len(of):
            mine[of] = shapely.contains_xy(shape, *xy[of].T)
    xy, col = xy[mine], col[mine]
    raised = np.all(col == PAVEMENT, axis=1) | np.all(col == KERB, axis=1) | np.all(col == PAVING_JOINT, axis=1)
    varied = col.copy()
    for colour in net.shapes:
        mine = np.all(col == colour, axis=1)
        varied[mine] = surface_colours(xy[mine], colour)
    pts = np.column_stack([xy[:, 0], -(ground(xy) + LIFT_M + raised * KERB_M), xy[:, 1]])
    curb_pts = []
    for quad in curb_faces(net, ground, net.all.envelope):
        a, b, _, d = quad
        s = max(.1, float(step(((a + b)[[0, 2]] / 2)[None])[0]))
        count = max(1, int(np.ceil(np.linalg.norm(b - a) / s)))
        q = a + ((np.arange(count) + .5) / count)[:, None] * (b - a) + .5 * (d - a)
        curb_pts.append(q[where(q[:, [0, 2]])])
    if curb_pts:
        curb_pts = np.concatenate(curb_pts)
        pts = np.r_[pts, curb_pts]
        varied = np.r_[varied, np.tile(np.array(KERB) * .82, (len(curb_pts), 1))]
    detail_pts, detail_cols = [], []
    for q, colour in street_details(net, ground):
        centre = q.mean(0)[[0, 2]]
        if not where(centre[None])[0]:
            continue
        s = max(.08, float(step(centre[None])[0]))
        samples = sample_quads(q[None], s)[0]
        detail_pts.append(samples)
        detail_cols.append(np.tile(colour, (len(samples), 1)))
    if detail_pts:
        pts = np.concatenate([pts, *detail_pts])
        varied = np.concatenate([varied, *detail_cols])
    return pts, varied
def _bridge_points(decks_, step):
    """(points, colours) of each deck and its side edges, step(xy) apart."""
    pts, cols = [], []
    for deck in decks_:
        s = float(step(deck.xy.mean(0)[None])[0])
        if s > deck.width:
            continue
        centre, h, side = deck.sample(s)
        across = np.arange(-deck.width / 2, deck.width / 2 + 1e-9, s)
        top = (centre[:, None] + across[None, :, None] * side[:, None]).reshape(-1, 2)
        pts.append(np.column_stack([top[:, 0], -np.repeat(h, len(across)), top[:, 1]]))
        cols.append(np.tile(deck.colour, (len(top), 1)))
        down = np.arange(s, DECK_M + 1e-9, s)
        for edge in (-deck.width / 2, deck.width / 2):
            e = centre + edge * side
            pts.append(np.column_stack([np.repeat(e[:, 0], len(down)),
                                        -(h[:, None] - down[None]).ravel(), np.repeat(e[:, 1], len(down))]))
            cols.append(np.tile(SIDE, (len(e) * len(down), 1)))
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(pts), np.concatenate(cols)
