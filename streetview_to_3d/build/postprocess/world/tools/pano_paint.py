"""Map points near the cameras coloured from the panos, as the fill does."""
import numpy as np

PAINT_M = 30.0                             # map points this near a camera are coloured from the panos


def pano_paint(views, pts, cols, scene, skip=None):
    """(cols, count): pts within PAINT_M of a camera recoloured from the panos (views: PanoViews)
    as the fill does (postprocess/tools/paint.py), except those marked skip (roofs)."""
    from scipy.spatial import cKDTree
    from streetview_to_3d.build.postprocess.tools.paint import paint
    cams, photos = views.cameras, views.photos
    if not cams or not len(pts):
        return cols, 0
    centres = np.array([c.centre for c in cams])
    near = cKDTree(centres[:, [0, 2]]).query(pts[:, [0, 2]], workers=-1)[0] < PAINT_M
    near = np.flatnonzero(near & ~skip if skip is not None else near)
    if not len(near):
        return cols, 0
    painted, who = paint(pts[near], scene, cams, [ph and (ph[0], ph[2]) for ph in photos], max_m=PAINT_M)
    cols = cols.copy()
    cols[near[who >= 0]] = painted[who >= 0]
    return cols, int((who >= 0).sum())
