"""Shared geo helpers: distances, flat-earth metres, parsing a location."""
import math
import re

import numpy as np

M_PER_DEG_LAT = 111320.0  # flat-earth: fine over one scene (hundreds of metres)


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def latlon_to_local_m(lat, lon, origin_lat, origin_lon):
    """(lat, lon) -> (east, north) metres from an origin, flat-earth."""
    m_per_lon = M_PER_DEG_LAT * math.cos(math.radians(origin_lat))
    return (lon - origin_lon) * m_per_lon, (lat - origin_lat) * M_PER_DEG_LAT


def local_m_to_latlon(east, north, origin_lat, origin_lon):
    """(east, north) metres from an origin -> (lat, lon), flat-earth; origin may be arrays."""
    return (origin_lat + north / M_PER_DEG_LAT,
            origin_lon + east / (M_PER_DEG_LAT * np.cos(np.radians(origin_lat))))


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
