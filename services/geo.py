"""Shared geo helpers -- used by app.py's single-pano flow and street_builder/."""
import math
import re


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def latlon_to_local_m(lat, lon, origin_lat, origin_lon):
    """(lat, lon) -> local ENU meters (east, north) relative to an origin --
    flat-earth approximation, fine over the scale of one reconstruction
    (hundreds of meters, not kilometers). Used to fit a segment's DA3-frame
    node positions against their real-world positions."""
    m_per_lat = 111320.0
    m_per_lon = 111320.0 * math.cos(math.radians(origin_lat))
    east = (lon - origin_lon) * m_per_lon
    north = (lat - origin_lat) * m_per_lat
    return east, north


def extract_lat_lon(raw: str):
    """Parse a Google Maps URL (.../@lat,lon,...) or a plain "lat,lon" string."""
    raw = raw.strip()
    m = re.search(r"@(-?\d+(?:\.\d+)?),(-?\d+(?:\.\d+)?)", raw)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)$", raw)
    if m:
        return float(m.group(1)), float(m.group(2))
    raise ValueError("Use a Google Maps URL with /@lat,lon or paste lat,lon directly.")
