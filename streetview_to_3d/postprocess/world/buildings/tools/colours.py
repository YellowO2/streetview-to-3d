"""Building colours softened from photos or livened from the satellite."""
import numpy as np

SOFT_LIGHT, SOFT_SAT = (0.62, 0.88), (0.15, 0.45)
CHEER_SAT, CHEER_LIFT = 1.3, 1.15


def soften(rgb):
    """rgb with lightness mapped into SOFT_LIGHT and saturation into SOFT_SAT, hue kept."""
    import colorsys
    h, l, s = colorsys.rgb_to_hls(*np.clip(rgb, 0, 1))
    l = SOFT_LIGHT[0] + (SOFT_LIGHT[1] - SOFT_LIGHT[0]) * l
    s = float(np.clip(s * 1.4, *SOFT_SAT))
    return np.array(colorsys.hls_to_rgb(h, l, s))


def cheer(rgb):
    """A dull satellite roof colour made CHEER_SAT more saturated and CHEER_LIFT brighter."""
    rgb = np.asarray(rgb, float)
    grey = rgb.mean(-1, keepdims=True)
    return np.clip((grey + (rgb - grey) * CHEER_SAT) * CHEER_LIFT, 0, 1)
