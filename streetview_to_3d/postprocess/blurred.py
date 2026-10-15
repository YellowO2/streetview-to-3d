"""Mark the blur hiding the capture rig at the bottom of a Google pano."""
import numpy as np
from PIL import Image
from scipy.ndimage import label, uniform_filter

BLUR_DETAIL, BLUR_WIN = 0.5, 15     # under this grey-level change per pixel, over BLUR_WIN px: blur
BLUR_GROW_DEG, SEAM_DEG = 6, 2


def blurred(path):
    """Mask of the smooth blur hiding the capture rig below the horizon in many Google panos.

    Low-detail regions (BLUR_DETAIL) connected to the bottom count, plus
    everything below them in each column, grown BLUR_GROW_DEG upward. The
    bottom SEAM_DEG rows are a hard seam and are skipped."""
    g = np.asarray(Image.open(path).convert("L"), float)
    h = g.shape[0]
    d = np.zeros_like(g)
    d[:, :-1] += np.abs(np.diff(g, axis=1))
    d[:-1] += np.abs(np.diff(g, axis=0))
    m = uniform_filter(d[:h - int(SEAM_DEG / 180 * h)], BLUR_WIN, mode=["nearest", "wrap"]) < BLUR_DETAIL
    m[:h // 2] = False
    lab, _ = label(m)
    m = np.isin(lab, np.unique(lab[-1][lab[-1] > 0]))
    top = np.where(m.any(0), m.argmax(0), h)
    # the blur reaches each column's bottom, so growing it moves its top up
    top = np.where(top < h, top - int(BLUR_GROW_DEG / 180 * h), h)
    return np.arange(h)[:, None] >= top[None, :]
