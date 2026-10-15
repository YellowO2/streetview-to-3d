"""Read the radius typed into the map's radius box."""


def typed_radius(radius_input):
    """The radius typed, in metres; None if it is not one yet."""
    try:
        radius_m = float(radius_input)
    except (TypeError, ValueError):
        return None
    return radius_m if radius_m > 0 else None
