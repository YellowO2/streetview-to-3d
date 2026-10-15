"""A bridge's deck: its centreline every metre and its height."""
import numpy as np

from streetview_to_3d.build_scene.postprocess.world.roads.tools.paving import LIFT_M

ARCH_PER, ARCH_MAX, DECK_M = 0.03, 2.0, 1.0     # bridge arch (share of length, max m); deck edge depth
LAYER_M, GRADE = 5.0, 0.06      # extra deck height per bridge layer; max deck slope
SIDE = np.array([0.60, 0.60, 0.58]) * 0.85      # bridge edge colour


class Deck:
    """A bridge deck: centreline every metre (xy, at) and its height h.

    Runs from the road height at one end to the other (ground), slightly
    arched, at least under(xy) plus LAYER_M per layer, no steeper than GRADE;
    pull(xy, h) snaps it to the scene where they touch. ids: its OSM ways."""

    def __init__(self, bridge, ground, under, pull=None):
        from scipy.ndimage import gaussian_filter1d
        xy, self.width, self.colour, layer, self.ids = bridge
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        cum = np.r_[0, np.cumsum(seg)]
        self.length = cum[-1]
        self.at = np.r_[np.arange(0, self.length, 1.0), self.length]
        self.xy = np.stack([np.interp(self.at, cum, xy[:, 0]), np.interp(self.at, cum, xy[:, 1])], 1)
        t = self.at / max(self.length, 1e-9)
        h0, h1 = ground(xy[[0, -1]]) + LIFT_M
        h = h0 + (h1 - h0) * t + min(ARCH_PER * self.length, ARCH_MAX) * 4 * t * (1 - t)
        h = np.maximum(h, under(self.xy) + LAYER_M * max(layer - 1, 0.0))
        h[0], h[-1] = h0, h1
        for order in (slice(None), slice(None, None, -1)):         # limit slope from both ends
            hh = h[order]
            for i in range(1, len(hh)):
                hh[i] = min(hh[i], hh[i - 1] + GRADE * 1.0)
        if pull is not None:
            h = pull(self.xy, h)
        if len(h) > 4:
            h[1:-1] = gaussian_filter1d(h, 2.0)[1:-1]
        self.h = h

    def sample(self, s):
        """(centreline every s m, its height, sideways unit vectors)."""
        at = np.r_[np.arange(0, self.length, s), self.length]
        centre = np.stack([np.interp(at, self.at, self.xy[:, 0]), np.interp(at, self.at, self.xy[:, 1])], 1)
        ahead = np.gradient(centre, axis=0) if len(centre) > 1 else np.array([[1.0, 0.0]])
        side = np.stack([-ahead[:, 1], ahead[:, 0]], 1) / np.maximum(np.linalg.norm(ahead, axis=1), 1e-9)[:, None]
        return centre, np.interp(at, self.at, self.h), side

    def footprint(self):
        import shapely
        return shapely.LineString(self.xy).buffer(self.width / 2, quad_segs=2, cap_style="flat")
