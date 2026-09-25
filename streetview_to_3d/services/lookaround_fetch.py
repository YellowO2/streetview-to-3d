"""Apple Look Around coverage lookup and panorama download."""
import io
import os

import pillow_heif
import torch as _torch
from PIL import Image
from streetlevel.lookaround import lookaround as apple_lookaround
from streetlevel.lookaround import reproject as apple_reproject
from streetlevel.lookaround.auth import Authenticator as AppleAuthenticator
from streetlevel.lookaround.reproject import to_equirectangular as apple_to_equirectangular

# ZeroGPU only permits CUDA ops inside @spaces.GPU-wrapped calls; reproject.py
# picks CUDA by default whenever it's "available", which ZeroGPU reports as
# true everywhere. Stitching a panorama doesn't need a GPU, so force CPU
# rather than spend GPU quota/allocation latency on it.
apple_reproject._device = _torch.device("cpu")
# Look Around faces are HEIC.
pillow_heif.register_heif_opener()

from streetview_to_3d.paths import PANOS_DIR

# Apple zoom is inverted vs Google's (0=full res/slow, 7=lowest). 3 is a
# fast, decent-quality default -- kept for whoever needs real detail.
APPLE_ZOOM = 3

# DA3 only (never SHARP appearance): DA3 caps each view slice at 504px,
# slice = pano_w/4. zoom=5 -> 2288px pano -> 572px slice, just above that
# cap -- measured directly. zoom=3's 4560px pano is 2x more than DA3 uses.
DA3_ONLY_APPLE_ZOOM = 5


_apple_auth = None


def get_apple_auth():
    global _apple_auth
    if _apple_auth is None:
        _apple_auth = AppleAuthenticator()
    return _apple_auth


def download_lookaround(pano, zoom: int = APPLE_ZOOM) -> str:
    """Fetch all 6 faces of a Look Around pano and stitch to equirectangular.
    Cached by ID + zoom -- a low-res (DA3-only) and high-res (SHARP
    appearance) request for the same pano must not collide."""
    img_path = os.path.join(PANOS_DIR, f"lookaround_{pano.id}_z{zoom}.jpg")
    if os.path.exists(img_path):
        return img_path

    auth = get_apple_auth()
    faces = [
        Image.open(io.BytesIO(apple_lookaround.get_panorama_face(pano, face_idx, zoom, auth)))
        for face_idx in range(6)
    ]
    equi = apple_to_equirectangular(faces, pano.camera_metadata)
    equi.save(img_path, quality=92)
    return img_path
