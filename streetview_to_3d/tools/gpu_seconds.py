"""The GPU window a walk over some dots needs, for the walk and the UI."""

# Headroom after the walk for saving the result before the window closes.
SAVE_BUFFER_S = 10.0
# DA3 is loaded at startup (streetview_to_3d.models.on_gpu), so a call only waits for ZeroGPU to move it.
MODEL_LOAD_S = 5.0

# How hard to try, as GPU seconds per spot: more time lets more capture dates patch weak stretches.
EFFORT_SECONDS_PER_SPOT = {"Quick": 4.0, "Normal": 8.0, "Thorough": 14.0}
DEFAULT_EFFORT = "Normal"


def estimate_gpu_seconds(n_dots: int, effort: str = DEFAULT_EFFORT) -> float:
    """The GPU window for a run over n_dots, model load and save headroom included.
    Kept no bigger than needed: it is checked against the user's ZeroGPU quota."""
    return MODEL_LOAD_S + n_dots * EFFORT_SECONDS_PER_SPOT[effort] + SAVE_BUFFER_S
