"""Google Street View: pano metadata and image downloads, cached under PANOS_DIR."""
import asyncio
import os

import aiohttp
from aiohttp import ClientSession, TCPConnector
from PIL import UnidentifiedImageError
from streetlevel import streetview

from streetview_to_3d.common.paths import PANOS_DIR
from streetview_to_3d.common.scene import node_key
from streetview_to_3d.common.http import BROWSER_HEADERS

# ~6656x3328 px: the default, for callers that need the photo's detail (the HF app's splat tab).
_DOWNLOAD_ZOOM = 4

# 2048 px wide: DA3 caps each view at 504 px and a view is a quarter of the pano, so more is wasted.
DA3_ONLY_ZOOM = 2


async def download_panorama_image(pano, img_path: str, zoom: int = _DOWNLOAD_ZOOM) -> None:
    """Download a panorama image with retry logic."""
    for attempt in range(4):
        try:
            # capped so a burst of tile requests doesn't trip Google's 403 rate limiter
            connector = TCPConnector(limit=10)
            async with ClientSession(headers=BROWSER_HEADERS, connector=connector) as dl_session:
                await streetview.download_panorama_async(pano, img_path, session=dl_session, zoom=zoom)
            return
        except (UnidentifiedImageError, Exception) as e:
            if attempt == 3:
                raise RuntimeError(f"Failed to download panorama after retries: {e}")
            wait = 3 ** attempt
            print(f"Tile fetch failed (attempt {attempt + 1}), retrying in {wait}s: {e}")
            await asyncio.sleep(wait)


def run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def format_date(d):
    if d is None:
        return "unknown date"
    s = f"{d.year:04d}-{d.month:02d}"
    if getattr(d, "day", None):
        s += f"-{d.day:02d}"
    return s


def google_node(pano_id, lat, lon, heading=None, **extra):
    """The node dict the map, the selection and the candidates all use for one Google pano."""
    return {"key": node_key("google", pano_id), "source": "google", "id": pano_id,
            "lat": lat, "lon": lon, "heading": heading, **extra}


def pano_to_meta(pano):
    """Metadata for a resolved StreetViewPanorama, however it was found."""
    neighbors = []
    for item in pano.links or pano.neighbors:
        n = item.pano if hasattr(item, "pano") else item
        if n and n.lat is not None:
            neighbors.append({"id": n.id, "lat": n.lat, "lon": n.lon})

    # every date is its own drive: its own position and heading
    dates = [{"id": p.id, "label": format_date(p.date), "lat": p.lat, "lon": p.lon,
              "heading": p.heading, "pitch": p.pitch, "roll": p.roll}
             for p in [pano, *(pano.historical or [])]]

    return {
        "id": pano.id,
        "lat": pano.lat,
        "lon": pano.lon,
        "date": format_date(pano.date),
        "elevation": pano.elevation,
        "neighbors": neighbors,
        "dates": dates,
        "heading": pano.heading,
        "pitch": pano.pitch,
        "roll": pano.roll,
    }


async def fetch_pano_by_id(pano_id):
    """Metadata for one panorama ID (e.g. a historical capture)."""
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        return pano_to_meta(pano)


async def fetch_panos_by_id(pano_ids, concurrency=16):
    """fetch_pano_by_id for many at once, in order; a failed one is None."""
    sem = asyncio.Semaphore(concurrency)

    async def one(pano_id):
        async with sem:
            try:
                return await fetch_pano_by_id(pano_id)
            except Exception as e:
                print(f"Pano lookup failed for {pano_id}: {e}")
                return None

    return await asyncio.gather(*(one(i) for i in pano_ids))


def _cache_path(pano_id, zoom):
    """Zoom is in the name, so two resolutions of one pano don't collide."""
    return os.path.join(PANOS_DIR, f"pano_{pano_id}_z{zoom}.jpg")


async def download_pano_by_id(pano_id, zoom: int = _DOWNLOAD_ZOOM):
    """Download a pano by its exact ID, return absolute path."""
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        img_path = _cache_path(pano.id, zoom)
        if os.path.exists(img_path):
            os.utime(img_path)  # in use again: keep paths.remove_older_than off it
        else:
            await download_panorama_image(pano, img_path, zoom=zoom)
        return img_path


def fetch_da3_pano(pano_id):
    """download_pano_by_id at DA3's resolution, blocking: the cached image's path, or None."""
    return run_async(download_pano_by_id(pano_id, zoom=DA3_ONLY_ZOOM))
