"""Run DA3 on some panos with this pipeline's view width, floor and mask.

The only caller of panoramic_da3.panoramic_da3. Each pano is segmented once, whole
(models/pano_labels.py), and each DA3 view cuts its part of that mask.
"""
import os
from functools import partial

import numpy as np

from streetview_to_3d.models.da3 import CONF_FLOOR, MASK_MOVERS, VIEW_HFOV
from streetview_to_3d.models.pano_labels import pano_labels
from streetview_to_3d.models.pano_mask import pano_mask


def run_da3(paths, cfg, da3, masker=None, mask_classes=None, **settings):
    """panoramic_da3 on the panos at `paths`, with this pipeline's view width, mask and
    confidence floor: (point_clouds, da3_result), each keyed by the pano's file name.
    masker, mask_classes: the segmenter repo and the classes to drop (None: defaults);
    `settings` are panoramic_da3.Settings fields (hfov, conf_floor: None for ours).
    Also used by the HF app."""
    from panoramic_da3 import Loader, Settings, panoramic_da3
    hfov, conf_floor = settings.pop("hfov", None), settings.pop("conf_floor", None)
    settings = Settings(
        da3_model=cfg.da3_model,
        hfov=hfov or VIEW_HFOV,
        conf_floor=CONF_FLOOR if conf_floor is None else conf_floor,
        drop_mask=_drop_mask(paths, masker, mask_classes),
        **settings,
    )
    return panoramic_da3(Loader(paths), settings, da3=da3)


def _drop_mask(paths, masker, mask_classes):
    if not MASK_MOVERS:
        return None
    panos = {os.path.basename(path): path for path in paths}
    return partial(_drop_in_views, panos=panos, model_id=masker, classes=mask_classes)


def _in_view(pano, yaw, hfov, w, h):
    """The part of a pano-shaped mask a w x h view at yaw sees (hfov wide,
    level, as panoramic_da3's Slicer cuts them)."""
    H, W = pano.shape
    f = w / 2 / np.tan(np.radians(hfov) / 2)
    u, v = np.meshgrid((np.arange(w) - (w - 1) / 2) / f, (np.arange(h) - (h - 1) / 2) / f)
    t = np.radians(yaw)
    x, z = np.cos(t) * u + np.sin(t), -np.sin(t) * u + np.cos(t)
    lon, lat = np.arctan2(x, z), np.arctan2(v, np.hypot(x, z))
    px = np.round((lon / (2 * np.pi) + .5) * (W - 1)).astype(int) % W
    py = np.clip(np.round((lat / np.pi + .5) * (H - 1)).astype(int), 0, H - 1)
    return pano[py, px]


def _drop_in_views(views, panos, model_id=None, classes=None, device=None):
    """One mask per DA3 view (panoramic_da3 Views; panos maps each view's
    pano_id to its file), cut from its pano's mask: each pano is segmented
    once, whole (pano_labels), and its class map kept."""
    masks, whole = [], {}
    for v in views:
        if v.pano_id not in whole:
            whole[v.pano_id] = pano_mask(pano_labels(panos[v.pano_id], model_id, device), classes)
        masks.append(_in_view(whole[v.pano_id], v.yaw, v.hfov, v.width, v.height))
    return masks
