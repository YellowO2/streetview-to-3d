"""Download a pano at the resolution DA3 uses, and wait for it."""
from streetview_to_3d.tools.run_async import run_async
from streetview_to_3d.tools.streetview_fetch.download_pano_by_id import DA3_ONLY_ZOOM, download_pano_by_id


def fetch_da3_pano(pano_id):
    """download_pano_by_id at DA3's resolution, blocking: the cached image's path, or None."""
    return run_async(download_pano_by_id(pano_id, zoom=DA3_ONLY_ZOOM))
