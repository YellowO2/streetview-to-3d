"""Draw what the masker drops from one pano, segmented on the Space over its API.

Segment on the Space (ui/pano_api.py's mask_pano), then draw locally with this
checkout's build/tools/pano_mask.py: every pole vs long straight poles only.

    python -m scripts.draw_mask PANO_ID OUT.jpg [--space potato-bug/street-view-to-3d-dev]
"""
import argparse
import base64
import io
import os

import numpy as np


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


def main():
    from gradio_client import Client
    from PIL import Image
    from streetview_to_3d.build.tools.pano_mask import pano_mask
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pano_id")
    ap.add_argument("out")
    ap.add_argument("--space", default="potato-bug/street-view-to-3d-dev")
    ap.add_argument("--masker", default="")
    a = ap.parse_args()
    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    got = np.load(io.BytesIO(base64.b64decode(Client(a.space, token=token).predict(a.pano_id, a.masker, api_name="/mask_pano"))))
    img = np.asarray(Image.open(io.BytesIO(got["pano"].tobytes())).convert("RGB"))
    ids = {str(n): i for i, n in enumerate(got["names"])}
    panels = [_overlay(img, pano_mask(got["labels"], long_only=long_only, label_ids=ids), text)
              for long_only, text in [(False, "every pole"), (True, "only long straight poles (what is used)")]]
    gap = np.full((8, img.shape[1], 3), 255, np.uint8)
    Image.fromarray(np.concatenate([panels[0], gap, panels[1]], 0)).save(a.out, quality=92)
    print("saved", a.out)


if __name__ == "__main__":
    main()
