"""Fetch many Street View panos' metadata at once."""
import asyncio
import logging

from streetview_to_3d.panos.tools.fetch_pano_by_id import fetch_pano_by_id

logger = logging.getLogger(__name__)


async def fetch_panos_by_id(pano_ids, concurrency=16):
    """fetch_pano_by_id for many at once, in order; a failed one is None."""
    sem = asyncio.Semaphore(concurrency)

    async def one(pano_id):
        async with sem:
            try:
                return await fetch_pano_by_id(pano_id)
            except Exception as e:
                logger.warning(f"Pano lookup failed for {pano_id}: {e}")
                return None

    return await asyncio.gather(*(one(i) for i in pano_ids))
