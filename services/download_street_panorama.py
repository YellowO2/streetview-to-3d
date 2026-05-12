import asyncio
import os

import numpy as np
from aiohttp import ClientSession
from PIL import UnidentifiedImageError
from streetlevel import streetview


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
