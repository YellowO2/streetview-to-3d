"""The ground's height round the scene, on Google's datum."""
import numpy as np

from streetview_to_3d.postprocess.world.WaterMap import SEA_M
from streetview_to_3d.postprocess.world.tools.seams import ramp

CELL_M = 25.0             # other panos' elevations: one per square this big
NEIGHBOUR_M, OUTLIER_M = 80.0, 2.5
OFF_GROUND_M = 15.0       # a pano this near a bridge or tunnel way is on it
PANOS_K, PANOS_SOFT_M = 8, 10.0            # panos the ground is interpolated from; distance softening
PANOS_FROM_M, PANOS_TO_M = 30.0, 150.0     # pano ground fades into the map between these distances


class Elevation:
    """The ground's height (m) at east/north points: the panos' elevations near them, the height
    map shifted onto Google's datum elsewhere (the sea at 0).

    site: Site (its panos' own elevations); heights: HeightMap; elevations: the other panos'
    (elevations_around), kept off bridges and tunnels (elements: OSM's), one per CELL_M square,
    without outliers, and not where the scene's own are. shift: the height map's datum
    correction, fixes: how far off it each pano is; known, more: how many panos of the scene's
    own and others' it stands on."""

    def __init__(self, site, heights, elevations, elements):
        from scipy.spatial import cKDTree
        self.site, self.heights = site, heights
        level = _on_ground(elevations.xy, elements, site.to_xy)
        more, more_el = _thinned(elevations.xy[level], elevations.metres[level])
        anchors, own_el = site.anchors, site.elevations
        if len(anchors) and len(more):
            keep = cKDTree(anchors).query(more)[0] > CELL_M
            more, more_el = more[keep], more_el[keep]
        at, elevation = np.r_[anchors, more], np.r_[own_el, more_el]
        self.fixes = elevation - heights(*site.to_ll(at)) if len(at) else np.zeros(0)
        self.shift = float(np.median(self.fixes)) if len(at) else 0.0
        self.known, self.more = len(anchors), len(more)
        self._panos = _from_panos(at, elevation)

    def __call__(self, xy, raw=None):
        """Heights at xy; raw: the height map's there, if already read."""
        raw = self.heights(*self.site.to_ll(xy)) if raw is None else raw
        h, w = self._panos(xy)
        return (np.where(raw <= SEA_M, 0.0, raw) + self.shift) * (1 - w) + h * w


def _from_panos(at, elevation):
    """f(east/north (n, 2)) -> (height, weight): pano elevations (at (m, 2)) spread by inverse distance,
    weight 1 within PANOS_FROM_M of the nearest, fading to 0 at PANOS_TO_M."""
    from scipy.spatial import cKDTree
    if not len(at):
        return lambda xy: (np.zeros(len(xy)), np.zeros(len(xy)))
    tree = cKDTree(at)

    def f(xy):
        k = min(PANOS_K, len(at))
        d, i = tree.query(xy, k=k)
        d, i = d.reshape(len(xy), k), i.reshape(len(xy), k)
        w = 1 / (d ** 2 + PANOS_SOFT_M ** 2)
        spread = (w * elevation[i]).sum(1) / w.sum(1)
        return spread, 1 - ramp((d[:, 0] - PANOS_FROM_M) / (PANOS_TO_M - PANOS_FROM_M))
    return f


def _on_ground(xy, elements, to_xy):
    """Whether each east/north xy (n, 2) is clear of OSM's bridges, tunnels and other non-ground roads."""
    import shapely
    off = [shapely.LineString(to_xy(e["geometry"])).buffer(OFF_GROUND_M)
           for e in elements if e.get("type") == "way" and "highway" in e.get("tags", {})
           and len(e.get("geometry") or []) >= 2 and _raised(e["tags"])]
    if not off or not len(xy):
        return np.ones(len(xy), bool)
    return ~shapely.contains_xy(shapely.union_all(off), xy[:, 0], xy[:, 1])


def _raised(tags):
    return (tags.get("bridge", "no") != "no" or tags.get("tunnel", "no") != "no"
            or tags.get("covered", "no") != "no" or tags.get("layer", "0") not in ("0", ""))


def _thinned(xy, elevation):
    """(xy (m, 2), elevation (m,)): one mean point and median elevation per CELL_M square,
    without squares more than OUTLIER_M off the plane of their neighbours within NEIGHBOUR_M."""
    from scipy.spatial import cKDTree
    if not len(xy):
        return np.zeros((0, 2)), np.zeros(0)
    key, inv = np.unique(np.floor(xy / CELL_M), axis=0, return_inverse=True)
    inv = inv.ravel()
    at = np.array([xy[inv == k].mean(0) for k in range(len(key))])
    h = np.array([np.median(elevation[inv == k]) for k in range(len(key))])
    keep = np.ones(len(at), bool)
    for k, near in enumerate(cKDTree(at).query_ball_point(at, NEIGHBOUR_M)):
        others = [j for j in near if j != k]
        if len(others) >= 3:
            rows = np.c_[np.ones(len(others)), at[others] - at[k]]
            plane = np.linalg.lstsq(rows, h[others], rcond=None)[0]
            keep[k] = abs(h[k] - plane[0]) <= OUTLIER_M
    return at[keep], h[keep]
