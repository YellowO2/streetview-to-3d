"""Tell Google's own Street View panos from the photos people add."""


def is_official(p):
    """Whether a streetlevel search result is one of Google's panos, not a user's photo sphere."""
    return p is not None and not (p.source or "").startswith("photos:")
