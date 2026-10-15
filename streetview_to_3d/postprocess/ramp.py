"""Ease from 0 to 1 smoothly, for blending one thing into another."""
import numpy as np


def ramp(t):
    """0 below 0, 1 above 1, smooth between."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)
