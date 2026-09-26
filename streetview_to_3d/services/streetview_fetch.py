"""Google Street View fetch/download logic for the single-pano flow in app.py."""
import asyncio
import os

import struct

import aiohttp
from aiohttp import ClientSession, TCPConnector
from PIL import UnidentifiedImageError
from streetlevel import streetview

from streetview_to_3d.paths import PANOS_DIR
from streetview_to_3d.services.http_headers import BROWSER_HEADERS

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

    # every date is its own capture -- a different drive, so its own position
    # and heading, metres and tens of degrees off the newest one's
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
    """Fetch pano metadata for a specific panorama ID (e.g. a historical capture)."""
    async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        return pano_to_meta(pano)


# Zoom baked into the cache filename -- a low-res (DA3-only) and high-res
# (SHARP appearance) request for the same pano must not collide.
def _cache_path(pano_id, zoom):
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




def _parse_depth_keeping_planes(b64):
    """streetlevel's depth parse, also keeping Google's per-pixel plane
    numbers, which it otherwise throws away. A depth map IS a list of planes
    plus one plane number per pixel (0 = sky), so with them every pixel's
    plane is known exactly instead of guessed back from the depth.
    None for a map that does not parse (some come back truncated), which
    streetlevel then treats as a pano with no depth."""
    import numpy as np
    from streetlevel.streetview import depth as sv_depth
    try:
        raw = sv_depth.decode_b64(b64)
        header = sv_depth.parse_header(raw)
        dm = sv_depth.parse(b64)
        dm.plane_index = np.asarray(sv_depth.parse_planes(header, raw)["indices"],
                                    np.int32).reshape(header["height"], header["width"])
    except (ValueError, IndexError, struct.error) as e:
        print(f"depth map did not parse ({e}) -- treated as none", flush=True)
        return None
    return dm


def _keep_depth_planes():
    from streetlevel.streetview import parse as sv_parse
    sv_parse.parse_depth = _parse_depth_keeping_planes


async def fetch_depth_planes(pano_id, session=None):
    """(depth, plane index) of a pano's Google depth map, both 256 x 512 and
    laid out like the photo, or None if it has none. depth is in metres
    along each ray, -1 for the sky; plane index is Google's plane number of
    each pixel, 0 for the sky.

    streetlevel mirrors its depth grid left-right against the photo; the
    plane numbers come in the photo's layout already, so only depth flips.
    """
    import numpy as np
    _keep_depth_planes()
    if session is None:
        async with aiohttp.ClientSession(headers=BROWSER_HEADERS) as s:
            pano = await streetview.find_panorama_by_id_async(pano_id, session=s, download_depth=True)
    else:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session, download_depth=True)
    if pano is None or pano.depth is None:
        return None
    return np.asarray(pano.depth.data, np.float32)[:, ::-1].copy(), pano.depth.plane_index.copy()


async def fetch_depth(pano_id):
    """Google's own depth map for a pano, in metres, laid out like the photo,
    or None if it has none. -1 marks the sky.

    Coarse: a flat plane per wall and one for the ground, no trees, cars or
    detail. But the ground is exact -- the camera comes out 2.4-2.5 m up --
    which is what reconstruct.ground_fill uses it for.
    """
    got = await fetch_depth_planes(pano_id)
    return None if got is None else got[0]
