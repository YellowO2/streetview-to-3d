"""Space a map point by how far it is from DA3's points.

The one rule for a map point's size: DA3's points are the smallest in the world; a map point is
as big on them, and bigger the further from them. Its spacing always follows its size.
"""
import numpy as np

DA3_M = 0.17              # DA3's points as drawn (the viewer's default look: 0.1 m x 1.4 x 1.2)
GROW = 0.0154             # a map point's size grows this much per metre from DA3's points
RATIO = 2.2               # a point's size, of its spacing: what leaves no gaps (the viewer's blocks.RATIO)
# A building's point goes only where DA3 has no point right there:
COVER = 0.75              # a map point with a DA3 point within COVER x its spacing is DA3's


def _size_at(d):
    """A map point's size (m) d metres from DA3's points."""
    return DA3_M + GROW * np.maximum(np.asarray(d, float), 0)


def spacing_at(d):
    """A map point's spacing (m) d metres from DA3's points: its size's."""
    return _size_at(d) / RATIO
