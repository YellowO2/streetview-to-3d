"""The masker alone, over the API: one pano's DA3 views, segmented on the
Space's GPU. For trying masker changes without a full reconstruction (and
without running the model on a laptop).

The Space returns each view's raw class map; everything after that
(which classes, long_poles, agree) is cheap and reruns locally from them:

    python -m streetview_to_3d.services.mask_api PANO_ID OUT.jpg [--space potato-bug/street-view-to-3d-dev]
"""
import base64
import io
import os
import sys
import tempfile

import numpy as np

from streetview_to_3d import gpu

GPU_SECONDS = 30


def views(pano_path, out_dir):
    """DA3's own views of a pano: same width, step and file names as a
    reconstruction's."""
    from panoramic_da3.components.ViewExtractor.ViewExtractor import extract_views_for_da3
    from streetview_to_3d.services.da3_ops import VIEW_HFOV, VIEW_STEP_DEGREES
    return extract_views_for_da3(pano_path, out_dir, step_degrees=VIEW_STEP_DEGREES,
                                 prefix="pano_0_", hfov=VIEW_HFOV)


def _label_task(paths, masker, pano=None):
    """The views' class maps and, given a pano path, the whole pano's too."""
    from streetview_to_3d.services.segment import label_views
    labels, ids = label_views(paths, model_id=masker or None)
    whole = label_views([pano], model_id=masker or None)[0][0] if pano else None
    return np.stack(labels), ids, whole


def _depth_task(paths, yaws, pano_id):
    from panoramic_da3.datatype import View
    from streetview_to_3d.services.da3_ops import VIEW_HFOV
    from streetview_to_3d.services.segment import label_views
    import math
    from PIL import Image
    w, h = Image.open(paths[0]).size
    f = w / 2 / math.tan(math.radians(VIEW_HFOV) / 2)
    vs = [View(yaw=y, pitch=0, path=p, width=w, height=h, focal_px=f, hfov=VIEW_HFOV,
               vfov=math.degrees(2 * math.atan(h / 2 / f)), pano_id=pano_id) for p, y in zip(paths, yaws)]
    kept, res = gpu.get_da3().process_views(vs, dist_thresh=0.2, angle_thresh=1)
    labels, ids = label_views(paths)
    pred = res.prediction
    return (np.stack(labels), ids, np.array([v.yaw for v in vs]),
            np.array([vs.index(v) for v in kept]), np.stack(pred.depth).astype(np.float16), np.stack(pred.conf).astype(np.float16))


def depth_pano(pano_id: str) -> str:
    """Base64 .npz of one Google pano's DA3 views run alone: as mask_pano,
    plus depth and conf (DA3's own, per kept view), kept (which views)."""
    from streetview_to_3d.services.da3_ops import VIEW_HFOV
    from streetview_to_3d.services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, run_async
    path = run_async(download_pano_by_id(pano_id, zoom=DA3_ONLY_ZOOM))
    vs = views(path, tempfile.mkdtemp())
    labels, ids, yaws, kept, depth, conf = gpu.run(_depth_task, [v.path for v in vs], [v.yaw for v in vs],
                                                   os.path.basename(path), seconds=90)
    names = [n for n, _ in sorted(ids.items(), key=lambda kv: kv[1])]
    buf = io.BytesIO()
    np.savez_compressed(buf, labels=labels, yaws=yaws, kept=kept, depth=depth, conf=conf, hfov=VIEW_HFOV,
                        names=np.array(names), pano=np.frombuffer(open(path, "rb").read(), np.uint8))
    return base64.b64encode(buf.getvalue()).decode()


def mask_pano(pano_id: str, masker: str = "") -> str:
    """Base64 .npz of one Google pano's DA3 views made tall (segment.tall_views),
    segmented: labels (views x h x w class ids), yaws, hfov, names (class
    names by id), pano (the photo's .jpg bytes)."""
    from streetview_to_3d.services.da3_ops import VIEW_HFOV
    from streetview_to_3d.services.segment import tall_views
    from streetview_to_3d.services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, run_async
    path = run_async(download_pano_by_id(pano_id, zoom=DA3_ONLY_ZOOM))
    tmp = tempfile.mkdtemp()
    vs = views(path, tmp)
    tall = tall_views(path, tmp, [v.yaw for v in vs], VIEW_HFOV, vs[0].width)
    labels, ids, whole = gpu.run(_label_task, tall, masker, path, seconds=GPU_SECONDS)
    names = [n for n, _ in sorted(ids.items(), key=lambda kv: kv[1])]
    buf = io.BytesIO()
    np.savez_compressed(buf, labels=labels, whole=whole, yaws=np.array([v.yaw for v in vs]), hfov=VIEW_HFOV,
                        names=np.array(names), pano=np.frombuffer(open(path, "rb").read(), np.uint8))
    return base64.b64encode(buf.getvalue()).decode()


def _overlay(img, drop, cov, text):
    """The whole pano, red where drop, dark where not cov; lines at DA3's
    reach (about 29 deg up and down) and at 70 deg down, as far as the
    fill colours."""
    from PIL import Image, ImageDraw
    img = img.astype(float)
    H, W = img.shape[:2]
    img[~cov] *= .35
    img[drop] = img[drop] * .45 + np.array([255, 0, 0]) * .55
    im = Image.fromarray(img.astype(np.uint8))
    dr = ImageDraw.Draw(im)
    for deg in (-29, 29, 70):
        y = int(H * (.5 + deg / 180))
        dr.line([0, y, W, y], fill=(255, 255, 0), width=1)
    dr.rectangle([0, 0, 7 * len(text) + 10, 20], fill=(0, 0, 0))
    dr.text((5, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def main(argv):
    """Segment on the Space, then draw locally with this checkout's
    segment.py: the tall views' mask (agreed, long straight poles) over
    the whole pano, against the whole pano segmented at once."""
    import argparse
    import cv2
    from gradio_client import Client
    from PIL import Image
    from streetview_to_3d.services import segment
    ap = argparse.ArgumentParser()
    ap.add_argument("pano_id")
    ap.add_argument("out")
    ap.add_argument("--space", default="potato-bug/street-view-to-3d-dev")
    ap.add_argument("--masker", default="")
    a = ap.parse_args(argv)
    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    got = np.load(io.BytesIO(base64.b64decode(Client(a.space, token=token).predict(a.pano_id, a.masker, api_name="/mask_pano"))))
    img = np.asarray(Image.open(io.BytesIO(got["pano"].tobytes())).convert("RGB"))
    yaws, hfov = got["yaws"], float(got["hfov"])
    ids = {str(n): i for i, n in enumerate(got["names"])}
    names = [f"pano_0_da3_{int(round(y))}_0.jpg" for y in yaws]
    masks = segment.agree(segment.masks_from_labels(list(got["labels"]), ids), names, hfov)
    panels = [_overlay(img, *segment.to_pano(masks, yaws, hfov, img.shape[:2]), "tall views, every view agrees (new)")]
    whole = cv2.resize(segment.masks_from_labels([got["whole"]], ids)[0].astype(np.uint8), img.shape[1::-1],
                       interpolation=cv2.INTER_NEAREST) > 0
    panels.append(_overlay(img, whole, np.ones_like(whole), "whole pano at once (what the fill used)"))
    gap = np.full((8, panels[0].shape[1], 3), 255, np.uint8)
    Image.fromarray(np.concatenate(sum([[p, gap] for p in panels], [])[:-1], 0)).save(a.out, quality=92)
    print("saved", a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
