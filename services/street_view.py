import asyncio
import itertools
import math
import os

import numpy as np
from aiohttp import ClientSession
from PIL import UnidentifiedImageError
from streetlevel import streetview
from streetlevel.dataclasses import Tile
from streetlevel.streetview import streetview as _sv_module

_NEW_TILE_URL = (
    "https://streetviewpixels-pa.googleapis.com/v1/tile"
    "?cb_client=maps_sv.tactile&panoid={0:}&x={1:}&y={2:}&zoom={3:}"
)


def _patched_generate_tile_list(pano, zoom):
    img_size = pano.image_sizes[zoom]
    cols = math.ceil(img_size.x / pano.tile_size.x)
    rows = math.ceil(img_size.y / pano.tile_size.y)
    return [
        Tile(x, y, _NEW_TILE_URL.format(pano.id, x, y, zoom))
        for x, y in itertools.product(range(cols), range(rows))
    ]


def patch_tile_url():
    _sv_module._generate_tile_list = _patched_generate_tile_list


async def download_panorama_image(pano, img_path: str) -> None:
    """Download a panorama image with retry logic."""
    for attempt in range(4):
        try:
            async with ClientSession() as dl_session:
                await streetview.download_panorama_async(pano, img_path, session=dl_session, zoom=5)
            return
        except (UnidentifiedImageError, Exception) as e:
            if attempt == 3:
                raise RuntimeError(f"Failed to download panorama after retries: {e}")
            wait = 3 ** attempt
            print(f"Tile fetch failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
            await asyncio.sleep(wait)


async def ensure_pano_downloaded(pano_id: str, session: ClientSession) -> str:
    """Ensure a panorama is cached in images/, downloading if needed. Returns the image path."""
    img_path = f"images/pano_{pano_id}.jpg"
    if os.path.exists(img_path):
        return img_path

    pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
    if not pano:
        raise ValueError(f"Panorama {pano_id} not found")

    os.makedirs("images", exist_ok=True)
    await download_panorama_image(pano, img_path)
    return img_path
