"""Mark what DA3 should leave out of a pano, from its class map.

Cars, people, water and long straight poles are dropped by default; any of
segment.CLASSES can be named per run instead.
"""
import cv2
import numpy as np
from scipy.ndimage import binary_dilation

from streetview_to_3d.models.segment import LABEL_IDS

MOVERS = ("person", "rider", "car", "truck", "bus", "motorcycle", "bicycle")   # not train: it is part of the place
THIN = ("pole", "traffic light", "traffic sign")
# What is dropped by default; any of CLASSES can be named per run instead.
DROP = MOVERS + THIN + ("water",)
# Grow each mask by a few pixels: depth at an object's edge smears between
# it and what's behind, and those in-between points are the worst floaters.
GROW_PX = 3
# A pole is dropped only as a long straight stick: POLE_LONG times taller than wide, and
# POLE_STRAIGHT of its rows within POLE_TOL widths of one line (bollards, pillars are kept).
POLE_LONG, POLE_STRAIGHT, POLE_TOL = 4.0, 0.8, 1.0


def pano_mask(labels, classes=None, long_only=True, label_ids=LABEL_IDS):
    """True on a class map's classes to drop (DROP by default; poles only where
    _long_poles says so, unless not long_only), grown by GROW_PX. label_ids:
    {class name: id} the map was labelled with."""
    unknown = set(classes or ()) - set(label_ids)
    if unknown:
        raise ValueError(f"not classes we label: {sorted(unknown)}")
    names = classes or DROP
    others = [label_ids[n] for n in names if n != "pole"]
    m = np.isin(labels, others)
    if "pole" in names:
        pole = labels == label_ids["pole"]
        m |= _long_poles(pole) if long_only else pole
    return binary_dilation(m, iterations=GROW_PX) if m.any() else m


def _long_poles(pole):
    """The parts of a pole mask that are long straight sticks (see POLE_LONG)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pole.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if h < 8:
            continue
        rows = lab[y:y + h, x:x + w] == k
        filled = rows.any(1)
        width = area / filled.sum()                       # its typical width per row
        if h < POLE_LONG * width:
            continue
        ys = np.flatnonzero(filled)
        cx = np.array([np.flatnonzero(r).mean() for r in rows[filled]])
        a, b = np.polyfit(ys, cx, 1)
        keep[k] = (np.abs(cx - (a * ys + b)) <= POLE_TOL * width).mean() >= POLE_STRAIGHT
    return keep[lab]
