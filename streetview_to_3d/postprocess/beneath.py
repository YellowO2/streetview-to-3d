"""Keep map ground just under the scene's own where both have ground."""
import numpy as np

# Where the map meets the scene's own points: its ground runs on a little way under the scene's,
# hiding that ground's ragged rim, and goes where the scene's covers (the land and Google's tiles
# alike).
UNDER_M = 0.1             # map ground kept this far under the scene's own
OVERLAP_M = 1.0           # ... for this far in from its edge; deeper in it is dropped


def beneath(h, ground, inside, overlap=OVERLAP_M):
    """(h, keep) for map ground at heights h where the scene may have ground of its own (ground:
    its height there; inside: metres in from its edge, 0 where it has none): kept UNDER_M
    beneath it within overlap of the edge, dropped deeper in."""
    return np.where(inside > 0, np.minimum(h, ground - UNDER_M), h), inside <= overlap
