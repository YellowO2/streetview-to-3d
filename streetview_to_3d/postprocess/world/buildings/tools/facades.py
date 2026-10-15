"""Where a building's windows go, and what colour they are."""
import numpy as np

from streetview_to_3d.postprocess.world.tools import styles
from streetview_to_3d.postprocess.world.tools.osm_tags import LEVEL_M, building_kind, tag_number

GLASS, GLASS_OWN = (0.22, 0.43, 0.64), 0.12   # window colour: blue glass plus a little of the wall's


def _number(tags, key):
    v = tag_number(tags, key)
    return v if v is not None and v >= 0 else None


WINDOW_U, WINDOW_V = (0.3, 0.7), (0.3, 0.8)     # window extent within a bay, within a floor


def facade_layout(form, h):
    """(bay, floor) metres for a building's windows, or None if it has none (styles.windows)."""
    many = styles.windows(form.tags)
    if many == "none":
        return None
    if many == "few":
        eaves = max(h - form.roof.height, 1.0)
        wall = max(eaves - form.base_m, 1.0)
        row = wall / np.ceil(wall / styles.FEW_ROW_M)
        return styles.FEW_BAY_M, float(eaves / max(1, round(eaves / row)))
    kind = building_kind(form.tags)
    if kind == "yes":
        kind = "office" if h >= 20 else "house" if h <= 10 and form.roof.poly.area < 180 else "yes"
    bay = 2.8 if kind in ("house", "detached", "terrace", "semidetached_house") else \
          2.4 if kind in ("office", "commercial") else \
          6.0 if kind in ("industrial", "warehouse", "shed", "garage", "garages") else 3.0
    levels = _number(form.tags, "building:levels")
    floor = np.clip((h - form.roof.height) / levels, 2.4, 5.0) if levels else LEVEL_M
    return bay, float(floor)


def glass(wall):
    """A window's colour in a wall of colour wall (..., 3)."""
    return np.clip(np.asarray(wall) * GLASS_OWN + GLASS, 0, 1)
