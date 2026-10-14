"""The masker (and DA3 depth) on one pano, on the Space's GPU over the API; drawn locally:

    python -m streetview_to_3d.models.mask_api PANO_ID OUT.jpg [--space potato-bug/street-view-to-3d-dev]
"""
import base64
import io
import os
import sys
import tempfile

import numpy as np

from streetview_to_3d.models import gpu

GPU_SECONDS = 30


def views(pano_path, out_dir):
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


def depth_pano(pano_id: str) -> str:
    """Base64 .npz of one Google pano's DA3 views run alone: as mask_pano,
    plus depth and conf (DA3's own, per kept view), kept (which views)."""
    from streetview_to_3d.models.da3 import VIEW_HFOV
    from streetview_to_3d.common.streetview_fetch import fetch_da3_pano
    path = fetch_da3_pano(pano_id)
    paths, yaws = views(path, tempfile.mkdtemp())
    labels, ids, yaws, kept, depth, conf = gpu.run(_depth_task, paths, yaws, os.path.basename(path), seconds=90)
    return _encode(path, ids, labels=labels, yaws=yaws, kept=kept, depth=depth, conf=conf, hfov=VIEW_HFOV)


def mask_pano(pano_id: str, masker: str = "") -> str:
    """Base64 .npz of one Google pano, segmented: labels (h x w class ids),
    names (class names by id), pano (the photo's .jpg bytes)."""
    from streetview_to_3d.common.streetview_fetch import fetch_da3_pano
    path = fetch_da3_pano(pano_id)
    labels, ids = gpu.run(_label_task, [path], masker, seconds=GPU_SECONDS)
    return _encode(path, ids, labels=labels[0])


def _overlay(img, drop, text):
    """The pano, red where drop; lines at DA3's reach (about 29 deg up and
    down) and at 70 deg down, as far as the fill colours."""
    from PIL import Image, ImageDraw
    img = img.astype(float)
    H = img.shape[0]
    img[drop] = img[drop] * .45 + np.array([255, 0, 0]) * .55
    im = Image.fromarray(img.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for deg in (-29, 29, 70):
        y = int(H * (.5 + deg / 180))
        dr.line([0, y, img.shape[1], y], fill=(255, 255, 0), width=1)
    dr.rectangle([0, 0, 7 * len(text) + 10, 20], fill=(0, 0, 0))
    dr.text((5, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def main(argv):
    """Segment on the Space, then draw locally with this checkout's
    segment.py: every pole vs long straight poles only."""
    import argparse
    from gradio_client import Client
    from PIL import Image
    from streetview_to_3d.models import segment
    ap = argparse.ArgumentParser()
    ap.add_argument("pano_id")
    ap.add_argument("out")
    ap.add_argument("--space", default="potato-bug/street-view-to-3d-dev")
    ap.add_argument("--masker", default="")
    a = ap.parse_args(argv)
    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    got = np.load(io.BytesIO(base64.b64decode(Client(a.space, token=token).predict(a.pano_id, a.masker, api_name="/mask_pano"))))
    img = np.asarray(Image.open(io.BytesIO(got["pano"].tobytes())).convert("RGB"))
    ids = {str(n): i for i, n in enumerate(got["names"])}
    panels = [_overlay(img, segment.masks_from_labels([got["labels"]], ids, long_only=long_only)[0], text)
              for long_only, text in [(False, "every pole"), (True, "only long straight poles (what is used)")]]
    gap = np.full((8, img.shape[1], 3), 255, np.uint8)
    Image.fromarray(np.concatenate([panels[0], gap, panels[1]], 0)).save(a.out, quality=92)
    print("saved", a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
