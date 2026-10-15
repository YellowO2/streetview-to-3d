"""Download a Street View pano's image by its ID, cached on disk."""
import asyncio
import logging
import os

from aiohttp import TCPConnector
from PIL import UnidentifiedImageError
from streetlevel import streetview

from streetview_to_3d.common.new_run_dir import PANOS_DIR
from streetview_to_3d.common.streetview_fetch.browser_session import browser_session

logger = logging.getLogger(__name__)

# ~6656x3328 px: the default, for callers that need the photo's detail (the HF app's splat tab).
_DOWNLOAD_ZOOM = 4

# 2048 px wide: DA3 caps each view at 504 px and a view is a quarter of the pano, so more is wasted.
DA3_ONLY_ZOOM = 2


async def _download_panorama_image(pano, img_path: str, zoom: int = _DOWNLOAD_ZOOM) -> None:
    """Download a panorama image with retry logic."""
    for attempt in range(4):
        try:
            # capped so a burst of tile requests doesn't trip Google's 403 rate limiter
            connector = TCPConnector(limit=10)
            async with browser_session(connector=connector) as dl_session:
                await streetview.download_panorama_async(pano, img_path, session=dl_session, zoom=zoom)
            return
        except (UnidentifiedImageError, Exception) as e:
            if attempt == 3:
                raise RuntimeError(f"Failed to download panorama after retries: {e}")
            wait = 3 ** attempt
            logger.warning(f"Tile fetch failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
            await asyncio.sleep(wait)


def _cache_path(pano_id, zoom):
    """Zoom is in the name, so two resolutions of one pano don't collide."""
    return os.path.join(PANOS_DIR, f"pano_{pano_id}_z{zoom}.jpg")


async def download_pano_by_id(pano_id, zoom: int = _DOWNLOAD_ZOOM):
    """Download a pano by its exact ID, return absolute path."""
    async with browser_session() as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        img_path = _cache_path(pano.id, zoom)
        if os.path.exists(img_path):
            os.utime(img_path)  # in use again: keep new_run_dir._remove_older_than off it
        else:
            await _download_panorama_image(pano, img_path, zoom=zoom)
        return img_path
