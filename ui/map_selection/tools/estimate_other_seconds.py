"""Guess how long a run takes off the GPU, for the user to plan by."""

# The CPU time around the window (downloads, then postprocess): mostly fixed, a little per spot.
OTHER_BASE_S, OTHER_SECONDS_PER_SPOT = 90.0, 4.0


def estimate_other_seconds(n_dots: int) -> float:
    """Roughly how long a run over n_dots takes off the GPU (no quota, only waiting)."""
    return OTHER_BASE_S + n_dots * OTHER_SECONDS_PER_SPOT
