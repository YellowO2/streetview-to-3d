"""Fetch one Street View pano's metadata: position, dates and links."""
from streetlevel import streetview

from streetview_to_3d.tools.streetview_fetch.browser_session import browser_session


def _format_date(d):
    if d is None:
        return "unknown date"
    s = f"{d.year:04d}-{d.month:02d}"
    if getattr(d, "day", None):
        s += f"-{d.day:02d}"
    return s


def _pano_to_meta(pano):
    """Metadata for a resolved StreetViewPanorama, however it was found."""
    neighbors = []
    for item in pano.links or pano.neighbors:
        n = item.pano if hasattr(item, "pano") else item
        if n and n.lat is not None:
            neighbors.append({"id": n.id, "lat": n.lat, "lon": n.lon})

    # every date is its own drive: its own position and heading
    dates = [{"id": p.id, "label": _format_date(p.date), "lat": p.lat, "lon": p.lon,
              "heading": p.heading, "pitch": p.pitch, "roll": p.roll}
             for p in [pano, *(pano.historical or [])]]

    return {
        "id": pano.id,
        "lat": pano.lat,
        "lon": pano.lon,
        "date": _format_date(pano.date),
        "elevation": pano.elevation,
        "neighbors": neighbors,
        "dates": dates,
        "heading": pano.heading,
        "pitch": pano.pitch,
        "roll": pano.roll,
    }


async def fetch_pano_by_id(pano_id):
    """Metadata for one panorama ID (e.g. a historical capture); dates labelled "YYYY-MM[-DD]"."""
    async with browser_session() as session:
        pano = await streetview.find_panorama_by_id_async(pano_id, session=session)
        if not pano:
            return None
        return _pano_to_meta(pano)
