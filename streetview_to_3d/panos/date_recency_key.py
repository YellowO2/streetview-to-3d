"""Order capture dates newest last, an unknown date oldest."""


def date_recency_key(date_str):
    """Sort key for fetch_pano_by_id's "YYYY-MM[-DD]" dates: "unknown date" sorts oldest."""
    return "" if date_str == "unknown date" else date_str
