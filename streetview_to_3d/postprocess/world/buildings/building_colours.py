"""Each building's colour: its own tag, else one from the panos' palette."""
import logging

import numpy as np

from streetview_to_3d.postprocess.world.buildings.tools.colours import soften
from streetview_to_3d.postprocess.world.tools.geometry import hash01

logger = logging.getLogger(__name__)

PALETTE_K, PALETTE_MIN, PALETTE_STRIDE = 8, 500, 8
PASTEL = [(0.95, 0.90, 0.80), (0.90, 0.72, 0.62), (0.74, 0.83, 0.92), (0.78, 0.90, 0.80),
          (0.94, 0.80, 0.80), (0.93, 0.88, 0.70), (0.82, 0.80, 0.90), (0.86, 0.86, 0.84)]


def building_colours(outlines, views):
    """(n, 3) a colour per outline (Outlines): its Form's, else one picked by place from the
    palette of the panos' building and wall pixels (views: PanoViews), else PASTEL."""
    if not outlines.items:
        return np.zeros((0, 3))
    try:
        palette = _palette(views.photos)
    except (OSError, ValueError) as e:
        logger.warning(f"terrain: no palette from the panos ({e!r})")
        palette = np.array(PASTEL), np.full(len(PASTEL), 1 / 8)
    return _colours(outlines.items, palette)


def _palette(photos):
    """(colours (k, 3), shares (k,)): PALETTE_K softened clusters of the panos' building and wall pixels,
    or PASTEL if they have under PALETTE_MIN. photos: (image, class map, mask) or None."""
    from PIL import Image
    from scipy.cluster.vq import kmeans2
    from streetview_to_3d.models.segment import LABEL_IDS
    ids = [LABEL_IDS["building"], LABEL_IDS["wall"]]
    samples = []
    for ph in photos:
        if ph is None:
            continue
        img = np.asarray(Image.open(ph[0]).convert("RGB"))
        h, w = img.shape[:2]
        v, u = np.mgrid[0:h:PALETTE_STRIDE, 0:w:PALETTE_STRIDE]
        at = lambda grid: grid[(v * grid.shape[0] // h).ravel(), (u * grid.shape[1] // w).ravel()]
        ok = np.isin(at(ph[1]), ids) & ~at(ph[2])
        samples.append(img[v.ravel(), u.ravel()][ok] / 255)
    samples = np.concatenate(samples) if samples else np.zeros((0, 3))
    if len(samples) < PALETTE_MIN:
        return np.array(PASTEL), np.full(len(PASTEL), 1 / len(PASTEL))
    centres, label = kmeans2(samples, PALETTE_K, seed=0, minit="++")
    share = np.bincount(label, minlength=PALETTE_K) / len(label)
    return np.array([soften(c) for c in centres]), share


def _colours(outlines, palette_):
    """(n, 3) building colours: Form.colour, else a palette colour picked by position, weighted by share."""
    cols, share = palette_
    cum = np.cumsum(share) / share.sum()
    out = np.empty((len(outlines), 3))
    for i, (xy, _, _, form, *_) in enumerate(outlines):
        tag = form.colour
        if tag is not None:
            out[i] = tag
            continue
        c = xy.mean(0)
        pick = hash01(c[0], c[1])
        out[i] = cols[min(int(np.searchsorted(cum, pick)), len(cols) - 1)]
    return out
