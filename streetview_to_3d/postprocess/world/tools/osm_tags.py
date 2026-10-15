"""What OSM's tags say of a thing's size and kind."""

LEVEL_M = 3.2             # a storey's height


def tag_number(tags, key):
    """A tag's leading number ("12 m", "3,5" -> 3.5), or None."""
    try:
        return float(str(tags[key]).split()[0].replace(",", "."))
    except (KeyError, ValueError, IndexError):
        return None


def building_kind(tags):
    """The building (or building:part) tag, "yes" if neither."""
    return tags.get("building", tags.get("building:part", "yes"))
