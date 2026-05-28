import asyncio
import os

import numpy as np
from aiohttp import ClientSession, TCPConnector
from PIL import UnidentifiedImageError
from streetlevel import streetview

# zoom=4 → ~6656×3328 px per pano, ~91 tiles. zoom=5 → ~13312×6656 px, ~338 tiles.
# Zoom 4 is high enough for the pipeline and keeps tile bursts well below Google's rate limit.
_DOWNLOAD_ZOOM = 4

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.google.com/maps/",
    "Accept-Language": "en-US,en;q=0.9",
}


async def download_panorama_image(pano, img_path: str, zoom: int = _DOWNLOAD_ZOOM) -> None:
    """Download a panorama image with retry logic."""
    for attempt in range(4):
        try:
            # TCPConnector limit caps concurrent tile connections so we don't burst
            # hundreds of requests at once and trigger Google's 403 rate limiter.
            connector = TCPConnector(limit=10)
            async with ClientSession(headers=_BROWSER_HEADERS, connector=connector) as dl_session:
                await streetview.download_panorama_async(pano, img_path, session=dl_session, zoom=zoom)
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
