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
# A road direction is only worth acting on when it is well determined.
# At a junction, or on a curve, the road is not a straight strip and its
# "direction" means little -- there the GPS-fitted heading is the better
# of the two and must be left alone.


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
