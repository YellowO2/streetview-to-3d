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


def _label_task(paths, masker):
    from streetview_to_3d.services.segment import label_views
    labels, ids = label_views(paths, model_id=masker or None)
    return np.stack(labels), ids


def mask_pano(pano_id: str, masker: str = "") -> str:
    """Base64 .npz of one Google pano's DA3 views, segmented: labels (views
    x h x w class ids), yaws, hfov, names (class names by id), pano (the
    photo's .jpg bytes)."""
    from streetview_to_3d.services.da3_ops import VIEW_HFOV
    from streetview_to_3d.services.streetview_fetch import DA3_ONLY_ZOOM, download_pano_by_id, run_async
    path = run_async(download_pano_by_id(pano_id, zoom=DA3_ONLY_ZOOM))
    vs = views(path, tempfile.mkdtemp())
    labels, ids = gpu.run(_label_task, [v.path for v in vs], masker, seconds=GPU_SECONDS)
    names = [n for n, _ in sorted(ids.items(), key=lambda kv: kv[1])]
    buf = io.BytesIO()
    np.savez_compressed(buf, labels=labels, yaws=np.array([v.yaw for v in vs]), hfov=VIEW_HFOV, names=np.array(names),
                        pano=np.frombuffer(open(path, "rb").read(), np.uint8))
    return base64.b64encode(buf.getvalue()).decode()


def _overlay(img, yaws, hfov, masks, text):
    """The pano's horizon band, red where each view drops in its own centre
    wedge (what DA3 turns into no points), dark where no view reaches."""
    from PIL import Image, ImageDraw
    img = img.astype(float)
    H, W = img.shape[:2]
    lon, lat = np.meshgrid((np.arange(W) / (W - 1) - .5) * 2 * np.pi, (np.arange(H) / (H - 1) - .5) * np.pi)
    d = np.stack([np.cos(lat) * np.sin(lon), np.sin(lat), np.cos(lat) * np.cos(lon)], -1)
    step = 360 / len(yaws)
    drop, cov = np.zeros((H, W), bool), np.zeros((H, W), bool)
    for yaw, m in zip(yaws, masks):
        h, w = m.shape
        f = w / 2 / np.tan(np.radians(hfov) / 2)
        t = np.radians(yaw)
        c = d @ np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])
        ok = c[..., 2] > 1e-6
        z = np.where(ok, c[..., 2], 1)
        x, y = f * c[..., 0] / z + (w - 1) / 2, f * c[..., 1] / z + (h - 1) / 2
        yo = np.degrees(np.arctan2(c[..., 0], c[..., 2]))
        ok &= (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1) & (abs(yo) <= step / 2)
        cov |= ok
        drop[ok] = m[np.round(y[ok]).astype(int), np.round(x[ok]).astype(int)]
    img[~cov] *= .35
    img[drop] = img[drop] * .45 + np.array([255, 0, 0]) * .55
    im = Image.fromarray(img.astype(np.uint8)[int(H * (.5 - 35 / 180)):int(H * (.5 + 35 / 180))])
    ImageDraw.Draw(im).rectangle([0, 0, 7 * len(text) + 10, 20], fill=(0, 0, 0))
    ImageDraw.Draw(im).text((5, 4), text, fill=(255, 255, 255))
    return np.asarray(im)


def main(argv):
    """Segment on the Space, then draw locally with this checkout's
    segment.py: every pole vs long straight poles only, both agreed."""
    import argparse
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
    panels = []
    for long_only, text in [(False, "every view agrees, every pole"),
                            (True, "every view agrees, only long straight poles (what DA3 uses)")]:
        masks = segment.masks_from_labels(list(got["labels"]), ids, long_only=long_only)
        panels.append(_overlay(img, yaws, hfov, segment.agree(masks, names, hfov), text))
    gap = np.full((8, panels[0].shape[1], 3), 255, np.uint8)
    Image.fromarray(np.concatenate([panels[0], gap, panels[1]], 0)).save(a.out, quality=92)
    print("saved", a.out)


if __name__ == "__main__":
    main(sys.argv[1:])
