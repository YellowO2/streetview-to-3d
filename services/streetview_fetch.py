"""Google Street View fetch/download logic for the single-pano flow in app.py."""
import asyncio
import os

import aiohttp
from aiohttp import ClientSession, TCPConnector
from PIL import UnidentifiedImageError
from streetlevel import streetview

from paths import IMAGES_DIR
from services.http_headers import BROWSER_HEADERS

# zoom=4 → ~6656×3328 px per pano, ~91 tiles. zoom=5 → ~13312×6656 px, ~338 tiles.
# Zoom 4 is high enough for SHARP (the actual 3DGS appearance source).
_DOWNLOAD_ZOOM = 4

# DA3 only (depth/pose, never SHARP appearance): DA3 internally caps each
# view slice at 504px regardless of input size, and a slice is pano_w/4.
# zoom=2 -> 2048px pano -> 512px slice, just above that cap -- measured
# directly, not estimated. Higher zoom here is wasted download+compute.
DA3_ONLY_ZOOM = 2


async def download_panorama_image(pano, img_path: str, zoom: int = _DOWNLOAD_ZOOM) -> None:
    """Download a panorama image with retry logic."""
    for attempt in range(4):
        try:
            # TCPConnector limit caps concurrent tile connections so we don't burst
            # hundreds of requests at once and trigger Google's 403 rate limiter.
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


def pano_to_meta(pano):
    """Shared metadata shape for a resolved StreetViewPanorama, however it was found."""
    neighbors = []
    for item in pano.links or pano.neighbors:
        n = item.pano if hasattr(item, "pano") else item
        if n and n.lat is not None:
            neighbors.append({"id": n.id, "lat": n.lat, "lon": n.lon})

    dates = [{"id": pano.id, "label": format_date(pano.date)}]
    for h in pano.historical or []:
        dates.append({"id": h.id, "label": format_date(h.date)})

    return {
        "id": pano.id,
        "lat": pano.lat,
        "lon": pano.lon,
        "date": format_date(pano.date),
        "neighbors": neighbors,
        "dates": dates,
        "heading": pano.heading,
        "pitch": pano.pitch,
        "roll": pano.roll,
    }


async def fetch_pano(lat, lon):
    """Fetch the newest pano at a location, with neighbor + historical-date stubs."""
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_async(lat, lon, session=session)
        if not pano:
            return None
        return pano_to_meta(pano)


async def fetch_pano_by_id(pano_id):
    """Fetch pano metadata for a specific panorama ID (e.g. a historical capture)."""
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        return pano_to_meta(pano)


# Zoom baked into the cache filename -- a low-res (DA3-only) and high-res
# (SHARP appearance) request for the same pano must not collide.
def _cache_path(pano_id, zoom):
    return os.path.join(IMAGES_DIR, f"pano_{pano_id}_z{zoom}.jpg")


async def download_pano(lat, lon, zoom: int = _DOWNLOAD_ZOOM):
    """Download a pano by lat/lon, return absolute path."""
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_async(lat, lon, session=session)
        if not pano:
            return None
        img_path = _cache_path(pano.id, zoom)
        if not os.path.exists(img_path):
            await download_panorama_image(pano, img_path, zoom=zoom)
        return img_path


async def download_pano_by_id(pano_id, zoom: int = _DOWNLOAD_ZOOM):
    """Download a pano by its exact ID, return absolute path."""
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        img_path = _cache_path(pano.id, zoom)
        if not os.path.exists(img_path):
            await download_panorama_image(pano, img_path, zoom=zoom)
        return img_path


async def download_images_for_nodes(nodes: list[dict], zoom: int = _DOWNLOAD_ZOOM) -> list[str]:
    """Download/cache images for an ordered list of {id, ...} node dicts (the
    shape both pano_to_meta's neighbor entries and street_builder's exported
    chain nodes share). The one place both the single-pano tab's support-pano
    gathering and the street-builder reconstruction script should go through,
    instead of each keeping a separate download implementation."""
    paths = []
    for i, node in enumerate(nodes):
        print(f"Downloading pano {i + 1}/{len(nodes)}: {node['id']}")
        path = await download_pano_by_id(node["id"], zoom=zoom)
        if not path:
            raise ValueError(f"Panorama {node['id']} not found")
        paths.append(path)
    return paths
