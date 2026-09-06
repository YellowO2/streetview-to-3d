"""Road alignment -- the refinement step that runs AFTER GPS alignment.

Pipeline position:  DA3 alignment -> GPS alignment -> ROAD ALIGNMENT

GPS places every piece roughly right, but leaves visible seams: pieces
sit at slightly different heights and slightly off sideways, and a piece
built from only one or two panoramas can be rotated almost arbitrarily
(one GPS point pins a position but says nothing about which way the
piece faces). This module fixes that by looking at the road itself.

The steps, in order:

  1. Look straight down at each piece -- for every ground cell keep the
     colour of its highest point. That is a plain top-down photo of the
     piece, no filtering.
  2. Grey cells are road. That gives each piece's road as a 2D shape.
  3. Measure which way that road runs. A road is long and thin, so its
     direction is the angle at which it is NARROWEST measured across.
     The painted white line is fitted separately as an independent check
     -- when the two agree, the direction is trustworthy.
  4. Decide whether this piece's heading may be touched, by comparing
     its road direction against its OWN GPS camera track:
       - they agree     -> GPS already got the heading right, leave it
       - they disagree  -> heading is wrong, solve for it
       - no track at all (single-node piece) -> heading was never
         constrained by anything, solve for it
     This self-check matters. Two neighbouring pieces can differ by
     several degrees simply because the road curves between them; that
     is real geometry, not error, and "correcting" it drags pieces
     metres away from their GPS positions.
  5. Slide the piece sideways ACROSS the road until its road best
     overlaps the reference's road. The kerbs give this a sharp answer.
  6. Shift it up or down until the two road surfaces sit at the same
     height.
  7. Leave the ALONG-road direction alone. A straight road looks
     identical at every point along its own length, so nothing in the
     imagery can determine it -- GPS keeps that one, permanently.

The blind spot directly beneath a panorama leaves a hole in the middle
of its road; every step here is written to tolerate it.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

CELL = 0.25              # metres per top-down cell for road work
ROAD_SAT_MAX = 0.16      # grey, as (max - min) / max: a RATIO, not a difference.
                         # Measured absolutely, a dark cell passes trivially --
                         # small numbers cannot differ much -- while a sunlit
                         # road of the same colour fails. Normalising by
                         # brightness is what chroma keying does, and it is the
                         # difference between the road being found in shadow and
                         # not.
ROAD_GREEN_MAX = 0.04    # ...and not vegetation. Grass is identified by its HUE
                         # (green above both red and blue) rather than by how
                         # saturated it is, so washed-out or shaded grass is
                         # still rejected.
ROAD_VAL_LO = 0.16       # ...and not pitch black
ROAD_VAL_HI = 0.85       # ...and not blown-out white (that is paint/sky)
RIDGE_TOL_CELLS = 1.0    # how far below the local maximum still counts as
                         # "down the middle"
MIN_ROAD_WIDTH_M = 2.5   # Footpaths and service lanes are the same grey as the
                         # road and no colour test will ever separate them. They
                         # are however NARROW, so they are removed by eroding
                         # until they vanish and growing back only what survives
                         # and still reaches the camera.
ROAD_MIN_AREA = 150      # cells; drops speckle, keeps road slabs
MIN_ROAD_HALF_WIDTH_M = 1.5   # a road is at least ~3 m across; a footpath is
                              # not. Measured on gap3: real road halves came
                              # out 1.68-2.57 m, the paths beside them 0.90 m
WHITE_PCT = 97.0         # white paint = brightest few % of on-road cells
WHITE_SAT_MAX = 0.10
HEADING_TOLERANCE_DEG = 3.0   # road vs own GPS track: within this = fine
# A road direction is only worth acting on when it is well determined.
# At a junction, or on a curve, the road is not a straight strip and its
# "direction" means little -- there the GPS-fitted heading is the better
# of the two and must be left alone.
MAX_SHARPNESS_DEG = 12.0      # spread of near-best angles; wider = ill-defined
MAX_WHITE_DISAGREE_DEG = 5.0  # shape and painted line must corroborate


def angdiff(a, b):
    """Signed difference between two undirected angles, in (-90, 90]."""
    return (a - b + 90) % 180 - 90


# --------------------------------------------------------------------
# step 1: top-down photo
# --------------------------------------------------------------------

def top_down(pts_xz, pts_y, cols, bounds, cell=CELL):
    """Straight-down view: per XZ cell keep the colour of the HIGHEST
    point in it. No filtering, no classification -- this is just what a
    camera directly overhead would see. Returns (img[nx,nz,3], occupied
    [nx,nz]), both indexed [ix, iz]."""
    min_x, max_x, min_z, max_z = bounds
    nx = max(int((max_x - min_x) / cell) + 1, 1)
    nz = max(int((max_z - min_z) / cell) + 1, 1)
    x, z = pts_xz[:, 0], pts_xz[:, 1]
    keep = (x >= min_x) & (x <= max_x) & (z >= min_z) & (z <= max_z)
    x, z, y, c = x[keep], z[keep], pts_y[keep], cols[keep]
    img = np.zeros((nx, nz, 3))
    occ = np.zeros((nx, nz), dtype=bool)
    if len(x) == 0:
        return img, occ
    ix = np.clip(((x - min_x) / cell).astype(int), 0, nx - 1)
    iz = np.clip(((z - min_z) / cell).astype(int), 0, nz - 1)
    flat = ix * nz + iz
    order = np.lexsort((-y, flat))          # highest point first per cell
    fs = flat[order]
    first = np.concatenate([[True], fs[1:] != fs[:-1]])
    top = order[first]
    tc = flat[top]
    img[tc // nz, tc % nz] = c[top]
    occ[tc // nz, tc % nz] = True
    return img, occ


def cell_centres(mask, bounds, cell=CELL):
    """World XZ of every set cell in a [nx,nz] mask."""
    min_x, _, min_z, _ = bounds
    ix, iz = np.nonzero(mask)
    return np.column_stack([min_x + ix * cell, min_z + iz * cell])


# --------------------------------------------------------------------
# step 2: road (and white paint) from the photo
# --------------------------------------------------------------------

def road_cells(pts_xz, pts_y, cols, bounds, cell=CELL, cams=None):
    """Road and white-paint cells for one piece.

    Three tests, in the order that matters:

      COLOUR  -- grey, mid-brightness, not green. Greyness is measured as a
                 ratio so it means the same thing in sun and in shadow.
      SHAPE   -- anything narrower than a road is removed, because a footpath
                 is exactly as grey as the road beside it.
      TRACK   -- of what is left, the road is what the camera drove along.

    Unobserved cells are bridged rather than treated as edges: the blind spot
    beneath a panorama is missing data, not the end of the road.

    `cams` is the piece's camera track as (N, 2) world XZ. Without it the
    colour and width tests still apply and every road-like surface is kept.

    White paint is thresholded per piece (brightest few percent of on-road
    cells) because exposure varies between panoramas.

    Returns (road_mask, road_xz, white_xz)."""
    img, occ = top_down(pts_xz, pts_y, cols, bounds, cell)
    mx, mn = img.max(2), img.min(2)
    sat = np.divide(mx - mn, np.maximum(mx, 1e-6))
    val = img.mean(2)
    green = img[:, :, 1] - np.maximum(img[:, :, 0], img[:, :, 2])
    grey = (occ & (sat <= ROAD_SAT_MAX) & (green <= ROAD_GREEN_MAX)
            & (val >= ROAD_VAL_LO) & (val <= ROAD_VAL_HI))
    road = _select_road(grey, occ, bounds, cell, cams)

    near = ndimage.binary_dilation(road, iterations=2) & occ
    if near.sum() < 20:
        white = np.zeros_like(road)
    else:
        white = near & (val >= np.percentile(val[near], WHITE_PCT)) & (sat <= WHITE_SAT_MAX)
    return road, cell_centres(road, bounds, cell), cell_centres(white, bounds, cell)


def _disc(radius_cells):
    n = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
    return (xx ** 2 + yy ** 2) <= radius_cells ** 2


def _select_road(grey, occ, bounds, cell, cams):
    """The road, out of everything road-coloured. See road_cells."""
    if not grey.any():
        return grey

    # bridge the blind spot, so a road split by it stays one surface
    hole = ndimage.binary_fill_holes(occ) & ~occ
    lab, n = ndimage.label(ndimage.binary_closing(grey | hole, np.ones((3, 3))))
    if n == 0:
        return np.zeros_like(grey)

    ci = None
    keep = set()
    if cams is not None:
        ci = np.column_stack([((cams[:, 0] - bounds[0]) / cell).astype(int),
                              ((cams[:, 1] - bounds[2]) / cell).astype(int)])
        keep = {int(lab[a, b]) for a, b in ci
                if 0 <= a < lab.shape[0] and 0 <= b < lab.shape[1] and lab[a, b]}
        # a region holding no observed road is the blind spot by itself
        keep = {c for c in keep if (grey & (lab == c)).any()}
    if not keep:
        sizes = ndimage.sum(grey, lab, range(1, n + 1))
        if sizes.max() < ROAD_MIN_AREA:
            return np.zeros_like(grey)
        keep = {int(np.argmax(sizes)) + 1}

    road = ndimage.binary_fill_holes(
        ndimage.binary_closing(np.isin(lab, sorted(keep)), np.ones((3, 3))))

    # Drop what is too narrow to be a road. Eroding removes thin spurs
    # outright; growing the survivor back inside the original mask returns
    # the road to full width without bringing the spurs with it.
    r = (MIN_ROAD_WIDTH_M / 2.0) / cell
    core = ndimage.binary_erosion(road, _disc(r))
    if ci is not None:
        lab2, n2 = ndimage.label(core)
        on = {int(lab2[a, b]) for a, b in ci
              if 0 <= a < lab2.shape[0] and 0 <= b < lab2.shape[1] and lab2[a, b]}
        if on:
            core = np.isin(lab2, sorted(on))
    if not core.any():
        return np.zeros_like(grey)
    for _ in range(int(r) + 3):
        core = ndimage.binary_dilation(core) & road
    return core


def centreline(road_mask, bounds, cell=CELL, min_frac=0.35):
    """Points down the middle of the road.

    The kerb is a poor thing to track: it is one edge of the road, so every
    error in the mask moves it, and a road bordered by unobserved space has
    no kerb there at all. The centre is the average of both edges, so the
    same errors largely cancel, and it survives one side being missing.

    It is the ridge of the distance-to-edge map -- the cells furthest from
    any edge, which is where the middle of a road is.
    """
    solid = ndimage.binary_fill_holes(
        ndimage.binary_closing(road_mask, np.ones((7, 7))))
    d = ndimage.distance_transform_edt(solid)
    if not solid.any() or d.max() <= 0:
        return np.zeros((0, 2))
    # How far from the edge the middle is depends on how wide this road is,
    # so the cut is a fraction of the widest point rather than a fixed
    # distance -- a fixed one silently returns nothing on a narrow road.
    # Keep the ridge PLATEAU, not just exact local maxima. Down the middle
    # of a straight road many cells are equally far from both kerbs, and
    # demanding an exact maximum picks a scatter of isolated cells out of
    # that plateau -- 32 points from a 5466-cell road, far too few to trace.
    ridge = solid & (d >= ndimage.maximum_filter(d, size=5) - RIDGE_TOL_CELLS) \
        & (d >= min_frac * d.max())
    ix, iz = np.nonzero(ridge)
    return np.column_stack([bounds[0] + ix * cell, bounds[2] + iz * cell])


# --------------------------------------------------------------------
# step 3: which way does the road run
# --------------------------------------------------------------------

def _width_at(pts_c, deg):
    t = np.radians(deg)
    w = pts_c @ np.array([-np.sin(t), np.cos(t)])
    # 2.5-97.5 percentile, not full extent: immune to stray cells and to
    # the blind-spot hole
    return float(np.percentile(w, 97.5) - np.percentile(w, 2.5))


def direction_from_shape(road_xz, step_deg=0.5):
    """The angle at which the road is narrowest across. Returns
    (deg, width_m, sharpness_deg) -- sharpness is the spread of angles
    within 15% of the best, so a small number means well determined."""
    q = road_xz - road_xz.mean(0)
    degs = np.arange(0, 180, step_deg)
    widths = np.array([_width_at(q, d) for d in degs])
    i = int(np.argmin(widths))
    ok = degs[widths <= widths[i] * 1.15]
    return float(degs[i]), float(widths[i]), float(ok.max() - ok.min())


def direction_from_white(white_xz, iters=400, tol=0.35, min_inliers=12, seed=0):
    """Straight line through the painted marking, by RANSAC -- the line
    is dashed and thin, so a plain PCA over all bright cells is easily
    pulled off by a bright kerb or a reflection. Returns
    (deg, length_m, n_inliers, n_candidates) or None."""
    if len(white_xz) < min_inliers:
        return None
    rng = np.random.default_rng(seed)
    best_n, best_inl = 0, None
    for _ in range(iters):
        a, b = white_xz[rng.choice(len(white_xz), 2, replace=False)]
        d = b - a
        n = np.linalg.norm(d)
        if n < 2.0:            # two adjacent pixels define nothing useful
            continue
        perp = np.array([-d[1] / n, d[0] / n])
        inl = np.abs((white_xz - a) @ perp) <= tol
        if inl.sum() > best_n:
            best_n, best_inl = int(inl.sum()), inl
    if best_inl is None or best_n < min_inliers:
        return None
    p = white_xz[best_inl]
    q = p - p.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    deg = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180)
    return deg, float(np.ptp(q @ vt[0])), best_n, len(white_xz)


def road_direction(road_xz, white_xz):
    """Combined estimate. Returns (deg, agreement_deg, sharpness_deg);
    agreement is None when there was not enough paint to fit. Both
    diagnostics small = trustworthy."""
    deg, _, sharp = direction_from_shape(road_xz)
    w = direction_from_white(white_xz)
    return deg, (None if w is None else angdiff(deg, w[0])), sharp


def gps_track_heading(cams_en):
    """Direction of a piece's own GPS camera track, or None for a piece
    with a single camera (which has no direction at all)."""
    if len(cams_en) < 2:
        return None
    q = cams_en - cams_en.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    return float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])) % 180)


# --------------------------------------------------------------------
# steps 4-6: solve the corrections
# --------------------------------------------------------------------

def direction_is_reliable(agree, sharp):
    """Is this road direction worth acting on? It needs to be sharply
    defined, and the shape and the painted line need to agree."""
    if sharp > MAX_SHARPNESS_DEG:
        return False, f"road direction ill-defined (+-{sharp/2:.0f} deg spread)"
    if agree is not None and abs(agree) > MAX_WHITE_DISAGREE_DEG:
        return False, f"shape and white line disagree by {agree:+.1f} deg"
    return True, ""


def heading_correction(road_xz, white_xz, cams_en, tol_deg=HEADING_TOLERANCE_DEG):
    """Decide whether this piece's heading may be touched, per step 4.

    Returns (free, own_deg, track_deg, reason). `free` True means the
    heading is NOT pinned by GPS and a caller should solve for it.

    A multi-node piece only gives up its GPS heading when the road
    disagrees AND the road direction is itself trustworthy -- otherwise
    a junction or a bend would be enough to throw away a heading that
    several GPS points had pinned down correctly."""
    own, agree, sharp = road_direction(road_xz, white_xz)
    track = gps_track_heading(cams_en)
    if track is None:
        return True, own, None, "single-node piece: GPS never constrained its heading"
    off = angdiff(own, track)
    if abs(off) <= tol_deg:
        return False, own, track, f"road matches own GPS track ({off:+.1f} deg): already correct"
    ok, why = direction_is_reliable(agree, sharp)
    if not ok:
        return False, own, track, f"road differs by {off:+.1f} deg but {why}: keeping GPS heading"
    return True, own, track, f"road disagrees with own GPS track by {off:+.1f} deg"


def solve_across(src_road_xz, dst_road_mask, bounds, across, cell=CELL,
                 range_m=6.0, step_m=0.1):
    """Step 5. Slide src sideways across the road, maximising how much of
    the two road shapes overlap. Returns ((offset_m, iou, n_cells),
    curve) -- the curve lets a caller see whether the peak is sharp."""
    min_x, max_x, min_z, max_z = bounds
    nx = int((max_x - min_x) / cell) + 1
    nz = int((max_z - min_z) / cell) + 1
    dst_n = int(dst_road_mask.sum())
    best = (0.0, -1.0, 0)
    curve = []
    for d in np.arange(-range_m, range_m + 1e-9, step_m):
        q = src_road_xz + d * across
        ix = ((q[:, 0] - min_x) / cell).astype(int)
        iz = ((q[:, 1] - min_z) / cell).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
        sm = np.zeros((nx, nz), bool)
        sm[ix[ok], iz[ok]] = True
        inter = int((sm & dst_road_mask).sum())
        union = int(sm.sum()) + dst_n - inter
        iou = inter / union if union else 0.0
        curve.append((d, iou))
        if iou > best[1]:
            best = (float(d), float(iou), inter)
    return best, np.array(curve)


def peak_width(curve, frac=0.9):
    """Spread of offsets scoring within `frac` of the best -- a narrow
    result means step 5 actually determined something."""
    best = curve[:, 1].max()
    near = curve[curve[:, 1] >= best * frac][:, 0]
    return float(near.min()), float(near.max())


def solve_height(src_xz, src_y, dst_xz, dst_y, max_d=0.3, min_points=100):
    """Step 6. Median height difference over points that land on top of
    each other once seen from above. Returns (offset_m, n_points)."""
    d, i = cKDTree(dst_xz).query(src_xz, distance_upper_bound=max_d)
    ok = np.isfinite(d)
    if ok.sum() < min_points:
        return 0.0, int(ok.sum())
    return float(np.median(dst_y[i[ok]] - src_y[ok])), int(ok.sum())


def local_road_axes(dst_road_xz, src_road_mask, dst_road_mask, bounds,
                    cell=CELL, radius_m=12.0):
    """Along/across unit vectors of the road WHERE the two pieces meet.
    Using the overlap rather than either piece as a whole matters on a
    curving road, where the two ends run at different angles. Returns
    (along, across, centre) or None if they do not overlap."""
    ov = src_road_mask & dst_road_mask
    if not ov.any():
        return None
    centre = cell_centres(ov, bounds, cell).mean(0)
    near = dst_road_xz[np.linalg.norm(dst_road_xz - centre, axis=1) < radius_m]
    if len(near) < 20:
        near = dst_road_xz
    q = near - near.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    along = vt[0]
    return along, np.array([-along[1], along[0]]), centre


def rotate_about(xz, deg, pivot):
    t = np.radians(deg)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (xz - pivot) @ R.T + pivot


# --------------------------------------------------------------------
# the whole thing
# --------------------------------------------------------------------

class RoadFix:
    """The correction for one piece: rotate `d_heading` about `pivot`,
    then shift `d_across` along `across`, then raise by `d_height`.
    Nothing moves along the road."""

    def __init__(self, d_heading, pivot, d_across, across, d_height, notes):
        self.d_heading = d_heading
        self.pivot = pivot
        self.across = across
        self.d_across = d_across
        self.d_height = d_height
        self.notes = notes

    def apply_xz(self, xz):
        out = xz if self.d_heading == 0.0 else rotate_about(xz, self.d_heading, self.pivot)
        return out + self.d_across * self.across

    def apply(self, xz, y):
        return self.apply_xz(xz), y + self.d_height

    def __repr__(self):
        return (f"RoadFix(heading={self.d_heading:+.2f}deg, "
                f"across={self.d_across:+.2f}m, height={self.d_height:+.2f}m)")


def align_piece(src, dst, src_cams, bounds, cell=CELL):
    """Road-align one piece to a reference piece.

    src/dst are each (xz, y, cols) already in shared GPS world coords.
    src_cams is src's GPS camera positions, used only for the step-4
    self-check and as the rotation pivot (rotating about the cameras
    keeps the GPS-known position exact).

    Returns (RoadFix, diagnostics dict)."""
    src_mask, src_road, src_white = road_cells(*src, bounds, cell)
    dst_mask, dst_road, dst_white = road_cells(*dst, bounds, cell)
    diag = {"src_road_cells": len(src_road), "dst_road_cells": len(dst_road),
            "overlap_cells_before": int((src_mask & dst_mask).sum())}
    notes = []

    free, own, track, reason = heading_correction(src_road, src_white, src_cams)
    diag.update(road_deg=own, track_deg=track, heading_free=free)
    notes.append(f"heading: {reason}")

    pivot = src_cams.mean(0)
    d_head = 0.0
    if free:
        solved, why = _solve_heading(src_road, own, dst_road, dst_mask,
                                     bounds, pivot, cell)
        if solved is None:
            notes.append(f"heading: left alone -- {why}")
        else:
            d_head = solved
            notes.append(f"heading: {d_head:+.2f} deg -- {why}")
            src_road = rotate_about(src_road, d_head, pivot)
            src_mask, _, _ = road_cells(rotate_about(src[0], d_head, pivot),
                                        src[1], src[2], bounds, cell)

    axes = local_road_axes(dst_road, src_mask, dst_mask, bounds, cell)
    if axes is None:
        notes.append("across/height: pieces' roads do not overlap -- nothing to solve")
        return RoadFix(d_head, pivot, 0.0, np.array([0.0, 0.0]), 0.0, notes), diag
    along, across, centre = axes
    diag["overlap_centre"] = centre

    (d_across, iou, inter), curve = solve_across(src_road, dst_mask, bounds, across, cell)
    lo, hi = peak_width(curve)
    diag.update(across_iou=iou, across_cells=inter, across_peak=(lo, hi))
    notes.append(f"across: {d_across:+.2f} m (IoU -> {iou:.3f}, peak {lo:+.1f}..{hi:+.1f} m)")

    moved_xz = rotate_about(src[0], d_head, pivot) + d_across * across
    d_height, n_pts = solve_height(moved_xz, src[1], dst[0], dst[1])
    diag["height_points"] = n_pts
    notes.append(f"height: {d_height:+.2f} m over {n_pts} overlapping point(s)")
    notes.append("along: not corrected -- GPS keeps it")

    return RoadFix(d_head, pivot, d_across, across, d_height, notes), diag


def reference_road_direction(dst_road_xz, near_xz, radius_m=20.0, min_cells=100):
    """Which way the reference's road runs in the neighbourhood of
    `near_xz` (normally the piece about to be placed). Local, not
    global: on a curving road the far end runs at a different angle and
    would give the wrong answer. Returns (deg, sharpness) or None."""
    near = dst_road_xz[np.linalg.norm(dst_road_xz - near_xz, axis=1) < radius_m]
    if len(near) < min_cells:
        return None
    deg, _, sharp = direction_from_shape(near)
    return deg, sharp


def _mask_iou(road_xz, dst_mask, bounds, cell):
    min_x, max_x, min_z, max_z = bounds
    nx = int((max_x - min_x) / cell) + 1
    nz = int((max_z - min_z) / cell) + 1
    ix = ((road_xz[:, 0] - min_x) / cell).astype(int)
    iz = ((road_xz[:, 1] - min_z) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iz >= 0) & (iz < nz)
    sm = np.zeros((nx, nz), bool)
    sm[ix[ok], iz[ok]] = True
    inter = int((sm & dst_mask).sum())
    union = int(sm.sum()) + int(dst_mask.sum()) - inter
    return (inter / union if union else 0.0), inter


def _solve_heading(src_road_xz, src_deg, dst_road_xz, dst_mask, bounds, pivot, cell):
    """Turn the piece so its road runs the same way the reference's road
    runs right there.

    Deliberately NOT a brute-force spin scored by overlap. Overlap is
    nearly flat with respect to rotation on a straight road -- measured
    at a couple of percent across twenty degrees -- so it cannot pick an
    angle out. The two road DIRECTIONS are each sharp to a few degrees,
    so matching them directly is far better conditioned. Overlap is used
    only to settle the one thing a direction cannot say: which of the
    two 180-degree-opposed choices is the right one.

    Returns (deg, reason), or (None, reason) if it could not be solved."""
    ref = reference_road_direction(dst_road_xz, pivot)
    if ref is None:
        return None, "no reference road nearby to take a direction from"
    ref_deg, ref_sharp = ref
    ok, why = direction_is_reliable(None, ref_sharp)
    if not ok:
        return None, f"reference {why}"
    cand = angdiff(ref_deg, src_deg)
    best = None
    for d in (cand, cand + 180.0):
        iou, _ = _mask_iou(rotate_about(src_road_xz, d, pivot), dst_mask, bounds, cell)
        if best is None or iou > best[1]:
            best = (d, iou)
    d = (best[0] + 180.0) % 360.0 - 180.0
    return d, (f"matched reference road {ref_deg:.1f} deg (+-{ref_sharp/2:.0f}) "
               f"to own {src_deg:.1f} deg")
