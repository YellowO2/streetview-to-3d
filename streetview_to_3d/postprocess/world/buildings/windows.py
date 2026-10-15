"""Glass painted on the walls of buildings without a physical facade."""
import numpy as np

from streetview_to_3d.postprocess.world.buildings.Blocks import SURFACE
from streetview_to_3d.postprocess.world.buildings.tools.facades import facade_layout, glass

WINDOW_U, WINDOW_V = (0.3, 0.7), (0.3, 0.8)     # window extent within a bay, within a floor




def windows(blocks, outlines, land):
    """blocks (Blocks) with the wall points that fall on windows glass (_windows); the walls
    stand on land (Land)."""
    _windows(blocks, outlines.items, land.surface)
    return blocks


def _windows(blocks, outlines, ground):
    """Paint wall points that fall on windows with glass, where points are dense enough
    (two per bay and floor); buildings with physical facades are skipped."""
    wall = np.flatnonzero((blocks.edge >= 0) & (blocks.kind == SURFACE))
    wall = wall[np.argsort(blocks.which[wall], kind="stable")]           # each building's together, once
    bounds = np.searchsorted(blocks.which[wall], np.arange(len(outlines) + 1))
    for owner, (xy, h, _, form, *_) in enumerate(outlines):
        i = wall[bounds[owner]:bounds[owner + 1]]
        if not len(i) or form.physical_facade:
            continue
        layout = facade_layout(form, h)
        if layout is None:
            continue
        bay, floor = layout
        up = blocks.v[i] - form.foot(xy, ground)
        qu, qv = blocks.u[i] / bay % 1, up / floor % 1
        on = ((qu >= WINDOW_U[0]) & (qu <= WINDOW_U[1]) & (qv >= WINDOW_V[0]) & (qv <= WINDOW_V[1])
              & (up >= 0.5) & (blocks.gap[i] * 2 <= min(bay, floor)))
        blocks.cols[i[on]] = glass(blocks.cols[i[on]])
