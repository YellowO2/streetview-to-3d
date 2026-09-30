"""A building's roof as OpenStreetMap shapes it ("roof:shape"), over its
outline, as points.

Two kinds, from any outline:

  - rising to one point (RADIAL: pyramidal, cone, dome, onion): the
    outline drawn again smaller and higher, ring above ring, to its
    middle -- spires and domes
  - one height over the whole outline (FIELD), across the outline's
    long axis (its smallest box round it): gabled (and saltbox,
    half-hipped), round, gambrel; from every edge: hipped, mansard; one
    slope down towards roof:direction (else across the short side):
    skillion

roof:orientation=across turns a ridge across the short side. A roof
without roof:height (or roof:levels) rises at SLOPE_DEG, a spire at 45
degrees, a dome as a half ball; any other shape is flat. Walls reach up
to it where it meets them (rise: a gable's triangle, a skillion's high
side). East/north metres, heights above the top of the walls.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon

SLOPE_DEG = 30.0
RADIAL = ("pyramidal", "cone", "dome", "onion")
ACROSS = ("gabled", "saltbox", "half-hipped", "round", "gambrel")
FROM_EDGES = ("hipped", "mansard")
FIELD = ACROSS + FROM_EDGES + ("skillion",)
ALIAS = {"cross_gabled": "hipped", "side_hipped": "hipped", "half_hipped": "half-hipped", "gable": "gabled",
         "hip": "hipped", "pyramid": "pyramidal", "spherical": "dome", "conical": "cone"}   # alike from afar
COMPASS = {k: 22.5 * i for i, k in enumerate(
    "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split())}
EPS_M = 0.05


def _profile(shape, u):
    """(height, size) of a RADIAL roof's ring u of the way up (0-1), both
    0-1: size 1 is the outline, 0 its middle."""
    if shape == "dome":
        return np.sin(u * np.pi / 2), np.cos(u * np.pi / 2)
    if shape == "onion":
        return u, (1 - u) ** 0.6 * (1 + 0.6 * np.sin(np.pi * u))
    return u, 1 - u


def _direction(text):
    """roof:direction, the way the roof faces (down its slope), in degrees
    from north, or None."""
    t = str(text).strip().upper()
    if t in COMPASS:
        return COMPASS[t]
    try:
        return float(t) % 360
    except ValueError:
        return None


def along(ring, step):
    """Points every step (about) round the closed ring (n, 2), its corners
    included."""
    out = []
    for a, c in zip(ring[:-1], ring[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(c - a) / step)))
        out.append(a + (c - a) * (np.arange(n) / n)[:, None])
    return np.concatenate(out) if out else np.zeros((0, 2))


class Roof:
    """The roof over outline xy (n, 2, closed): shape, height (None:
    guessed, see default), direction (roof:direction's text), across
    (roof:orientation=across)."""

    def __init__(self, xy, shape, height=None, direction=None, across=False):
        self.xy, self.args = xy, (shape, direction, across)
        self.poly = Polygon(xy).buffer(0)
        shape = ALIAS.get(shape, shape)
        self.shape = shape if shape in RADIAL + FIELD and self.poly.area > 1 else "flat"
        if self.shape == "flat":
            self.height = 0.0
            return
        with np.errstate(invalid="ignore"):                  # GEOS warns on a few thin outlines
            box = np.asarray(self.poly.minimum_rotated_rectangle.exterior.coords)[:4]
        if len(box) < 4 or not np.isfinite(box).all():
            (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
            box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        sides = [box[1] - box[0], box[2] - box[1]]
        long_, short = sorted(sides, key=np.linalg.norm, reverse=True)
        if across:
            long_, short = short, long_
        self.axis = long_ / np.linalg.norm(long_)           # the ridge runs this way
        self.half = np.linalg.norm(short) / 2               # eave to ridge
        self.centre = box.mean(0)
        self.middle = np.asarray(self.poly.centroid.coords[0])
        self.radius = float(np.linalg.norm(xy[:-1] - self.middle, axis=1).mean())
        d = _direction(direction) if direction is not None else None
        self.down = np.array([np.sin(np.radians(d)), np.cos(np.radians(d))]) if d is not None \
            else np.array([-self.axis[1], self.axis[0]])
        h = xy[:-1] @ self.down
        self.lo_hi = (h.min(), h.max())
        self.height = self.default() if height is None else float(height)

    def on(self, xy):
        """The same roof, as high, over another outline (the building moved
        or cut)."""
        return Roof(xy, *self.args[:1], self.height, *self.args[1:])

    def default(self):
        """A roof:height for a roof that has none."""
        if self.shape in RADIAL:
            return self.radius                                # a spire at 45 degrees, a dome a half ball
        if self.shape == "skillion":
            return np.tan(np.radians(SLOPE_DEG)) * (self.lo_hi[1] - self.lo_hi[0])
        return np.tan(np.radians(SLOPE_DEG)) * self.half

    def _field(self, en):
        """0-1, a FIELD roof's height at east/north points."""
        if self.shape == "skillion":
            lo, hi = self.lo_hi
            return np.clip((hi - en @ self.down) / max(hi - lo, 1e-6), 0, 1)
        if self.shape in FROM_EDGES:
            d = shapely.distance(shapely.points(en), self.poly.exterior) / max(self.half, 1e-6)
            d = np.clip(d, 0, 1)
            return np.interp(d, [0, .3, 1], [0, .75, 1]) if self.shape == "mansard" else d
        a = np.clip(np.abs((en - self.centre) @ np.array([-self.axis[1], self.axis[0]])) / max(self.half, 1e-6),
                    0, 1)                                     # 0 on the ridge, 1 at the eaves
        if self.shape == "round":
            return np.sqrt(1 - a ** 2)
        if self.shape == "gambrel":
            return np.interp(a, [0, .6, 1], [1, .75, 0])
        return 1 - a

    def rise(self, en):
        """How far over the walls' top the roof stands at east/north points
        on its outline: what a wall there reaches up to."""
        if self.shape in FIELD and self.height > 0 and self.shape not in FROM_EDGES:
            return self.height * self._field(en)
        return np.zeros(len(en))

    def surface(self, step):
        """(points (m, 3) east, north, up over the walls' top; their
        normals (m, 3), the same way round, facing out), about step
        apart."""
        if self.shape in RADIAL and self.height > 0:
            return self._rings(step)
        lo, hi = self.xy.min(0), self.xy.max(0)
        slope = self.height / max(self.half, 1e-6) if self.shape in FIELD and self.height > 0 else 0.0
        s = step / max(1.0, min(3.0, np.hypot(1, slope)))    # a steep roof as dense along its slope
        gx, gy = np.meshgrid(np.arange(lo[0], hi[0], s) + s / 2, np.arange(lo[1], hi[1], s) + s / 2)
        en = np.stack([gx.ravel(), gy.ravel()], 1)
        en = np.concatenate([en[shapely.contains_xy(self.poly, *en.T)], self.xy[:-1]])
        if self.shape == "flat" or self.height <= 0:
            return np.c_[en, np.zeros(len(en))], np.tile([0.0, 0.0, 1.0], (len(en), 1))
        z = self.height * self._field(en)
        gx = (self.height * self._field(en + [EPS_M, 0]) - z) / EPS_M
        gy = (self.height * self._field(en + [0, EPS_M]) - z) / EPS_M
        n = np.c_[-gx, -gy, np.ones(len(en))]
        return np.c_[en, z], n / np.linalg.norm(n, axis=1, keepdims=True)

    def _rings(self, step):
        u = np.linspace(0, 1, 400)
        zn, size = _profile(self.shape, u)
        r, z = size * self.radius, zn * self.height
        arc = np.r_[0, np.cumsum(np.hypot(np.diff(r), np.diff(z)))]
        pick = np.searchsorted(arc, np.arange(0, arc[-1], step))
        pts, normals = [], []
        for k in pick:
            ring = self.middle + (self.xy - self.middle) * size[k]
            en = along(ring, step) if size[k] > 1e-3 else self.middle[None]
            j = min(k + 1, len(u) - 1), max(k - 1, 0)
            dr, dz = r[j[0]] - r[j[1]], z[j[0]] - z[j[1]]
            out = en - self.middle
            out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
            n = np.c_[out * dz, np.full(len(en), -dr)]
            normals.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9))
            pts.append(np.c_[en, np.full(len(en), z[k])])
        pts.append(np.r_[self.middle, self.height][None])
        normals.append(np.array([[0.0, 0.0, 1.0]]))
        return np.concatenate(pts), np.concatenate(normals)
