"""Keep a dot's closest panos of one capture date (no GPU)."""
from streetview_to_3d.common.geo import haversine_m

# Per dot, per date, how many of the closest panos to keep (and so download and rate):
# enough for a fallback when the closest is bad; more is the same spot again.
TOP_PANOS_PER_DOT = 3


def cap_bucket_for_date(bucket, date, dot_lat, dot_lon, top_n):
    """This dot's own panos of ONE date, closest-first, capped to top_n."""
    same_date = [n for n in bucket if n["date"] == date]
    same_date.sort(key=lambda n: haversine_m(dot_lat, dot_lon, n["lat"], n["lon"]))
    return same_date[:top_n]
