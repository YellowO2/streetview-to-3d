"""Serve the masker and DA3 on one pano over the Space's API.

Two API-only endpoints, mask_pano and depth_pano, each run on the Space's GPU; the
masker's drawing over the API is `python -m streetview_to_3d.models`.
"""
import base64
import io
import os
import tempfile

import gradio as gr
import numpy as np

from streetview_to_3d.models import gpu

GPU_SECONDS = 30


def _views(pano_path, out_dir):
    """DA3's own views of a pano, as a reconstruction cuts them, written to out_dir
    for the segmenter: (paths, yaws)."""
    import cv2
    from panoramic_da3 import Loader, Settings, ViewExtractor
    from streetview_to_3d.models.da3 import VIEW_HFOV, VIEW_STEP_DEGREES
    cut = ViewExtractor(Loader([pano_path]), Settings(da3_model="", step_degrees=VIEW_STEP_DEGREES, hfov=VIEW_HFOV))
    paths = [os.path.join(out_dir, f"pano_0_da3_{int(round(v.yaw))}_0.jpg") for v in cut]
    for path, v in zip(paths, cut):
        cv2.imwrite(path, v.image)
    return paths, [v.yaw for v in cut]


def _label_task(paths, masker):
    from streetview_to_3d.models.segment import label_views
    labels, ids = label_views(paths, model_id=masker or None)
    return np.stack(labels), ids


def _encode(path, ids, **arrays):
    """Base64 .npz of arrays, plus names (class names by id) and pano (the photo's .jpg bytes)."""
    names = [n for n, _ in sorted(ids.items(), key=lambda kv: kv[1])]
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays, names=np.array(names),
                        pano=np.frombuffer(open(path, "rb").read(), np.uint8))
    return base64.b64encode(buf.getvalue()).decode()


def _depth_task(paths, yaws, pano_id):
    import cv2
    from panoramic_da3 import View
    from streetview_to_3d.models.da3 import VIEW_HFOV
    from streetview_to_3d.models.segment import label_views
    import math
    images = [cv2.imread(p) for p in paths]
    h, w = images[0].shape[:2]
    f = w / 2 / math.tan(math.radians(VIEW_HFOV) / 2)
    vs = [View(image=im, yaw=y, pitch=0, width=w, height=h, focal_px=f, hfov=VIEW_HFOV,
               vfov=math.degrees(2 * math.atan(h / 2 / f)), pano_id=pano_id) for im, y in zip(images, yaws)]
    res = gpu.get_da3().process_views(vs, dist_thresh=0.2, angle_thresh=1)
    labels, ids = label_views(paths)
    pred = res.prediction
    kept = [i for i, v in enumerate(vs) if any(v is k for k in res.views)]
    return (np.stack(labels), ids, np.array([v.yaw for v in vs]),
            np.array(kept), np.stack(pred.depth).astype(np.float16), np.stack(pred.conf).astype(np.float16))


def _depth_pano(pano_id: str) -> str:
    """Base64 .npz of one Google pano's DA3 views run alone: as mask_pano,
    plus depth and conf (DA3's own, per kept view), kept (which views)."""
    from streetview_to_3d.models.da3 import VIEW_HFOV
    from streetview_to_3d.common.streetview_fetch import fetch_da3_pano
    path = fetch_da3_pano(pano_id)
    paths, yaws = _views(path, tempfile.mkdtemp())
    labels, ids, yaws, kept, depth, conf = gpu.run(_depth_task, paths, yaws, os.path.basename(path), seconds=90)
    return _encode(path, ids, labels=labels, yaws=yaws, kept=kept, depth=depth, conf=conf, hfov=VIEW_HFOV)


def _mask_pano(pano_id: str, masker: str = "") -> str:
    """Base64 .npz of one Google pano, segmented: labels (h x w class ids),
    names (class names by id), pano (the photo's .jpg bytes)."""
    from streetview_to_3d.common.streetview_fetch import fetch_da3_pano
    path = fetch_da3_pano(pano_id)
    labels, ids = gpu.run(_label_task, [path], masker, seconds=GPU_SECONDS)
    return _encode(path, ids, labels=labels[0])


def pano_api():
    """Register the API-only endpoints mask_pano and depth_pano, inside a gr.Blocks."""
    gr.api(_mask_pano, api_name="mask_pano")
    gr.api(_depth_pano, api_name="depth_pano")
