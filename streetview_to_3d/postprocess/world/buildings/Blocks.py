"""Buildings as points: roofs, walls and details, with what each point is."""
import numpy as np

ROOF, INNER = -1, -2          # Blocks.edge for roof and inner-wall points
SURFACE, EDGE = 0, 1          # Blocks.kind, as effects/blocks.js draws them


class Blocks:
    """Every building's points with, per point: building (which), wall edge (edge; ROOF, INNER),
    u metres along and v metres up its wall, spacing (gap), tag colour kept (own), normal and kind
    (SURFACE or EDGE). edges[e]: (start, along, length, DA3 plane (n2, d2) or None).

    As the buildings machine goes on: reached (one per building), whether DA3 reaches it; seen,
    how many took the panos' colour; cut, how many points were left to DA3; near (per point,
    seams.toward) and painted, how many the panos coloured."""

    def __init__(self, pts, cols, which, edge, u, v, gap, edges, own, normal, kind):
        self.pts, self.cols, self.which = pts, cols, which
        self.edge, self.u, self.v, self.gap, self.edges = edge, u, v, gap, edges
        self.own, self.normal, self.kind = own, normal, kind
        self.reached, self.near = np.zeros(0, bool), np.zeros(len(pts))
        self.seen = self.cut = self.painted = 0

    def take(self, keep):
        for k in ("pts", "cols", "which", "edge", "u", "v", "gap", "own", "normal", "kind"):
            setattr(self, k, getattr(self, k)[keep])
