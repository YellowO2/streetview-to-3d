"""Pull the road surface and its edges out of a piece.

The road is found as the connected patch of grey ground the piece's own
cameras stand on, then reduced to a top-down mask. The kerb comes back as
a THIN line -- one point per cell along the actual road/not-road boundary.

Holes inside the road are filled before taking that boundary: an
unobserved patch in the middle of the tarmac is a gap in the data, not a
kerb, and without filling it produced edges running down the centre of
the road.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from postprocess.road_align.road_mask import CELL, road_cells, top_down

SURFACE_BAND_M = 0.75    # points further than this from the road surface are
                         # grass hanging below the kerb, or things standing on
                         # the road -- measured, not guessed: a quarter of the
                         # points inside road cells were foliage below kerb level
KERB_NEIGHBOURS = 8      # points used to estimate the kerb's local direction
ROAD_NEIGHBOURS = 12     # points used to estimate the road's local normal
DEGENERATE_EIGENVALUE = 1e-3   # below this a direction is not constrained at all


def extract_features(piece, bounds, cell=CELL, band=SURFACE_BAND_M, cams=None):
    """(road_xyz, kerb_xyz, paint_xyz) for one piece.

    The kerb comes out as a THIN line -- one point per cell along the
    actual road/not-road boundary. An earlier version dilated the
    boundary and kept every point inside those cells, which turned the
    kerb into a blob covering a quarter of the road. Blobs register
    against blobs at any offset, which defeats the whole point of using
    an edge feature.

    Holes inside the road are filled before taking the boundary. An
    unobserved patch in the middle of the tarmac is a gap in the data,
    not a kerb, and it produced edges running down the centre of the
    road.

    Pass `cams` -- the piece's camera track -- so road_cells can tell the
    road from the footpaths and forecourts beside it. Without it every
    grey surface wide enough to qualify contributes its own boundary, and
    the kerb comes back as many short unrelated fragments."""
    xz, y, cols = piece
    road_mask, _, paint_xz = road_cells(xz, y, cols, bounds, cell, cams=cams)
    _, occ = top_down(xz, y, cols, bounds, cell)

    solid = ndimage.binary_closing(road_mask, np.ones((5, 5)))
    solid = ndimage.binary_fill_holes(solid)
    ring = ndimage.binary_dilation(solid, iterations=1) & ~solid
    kerb_mask = ndimage.binary_dilation(ring & occ, iterations=1) & solid
    # the outer rim, where the panorama ran out of range, is not a kerb
    kerb_mask &= ~(ndimage.binary_dilation(ring & ~occ, iterations=1) & solid)
    kerb_mask &= road_mask

    nx, nz = road_mask.shape
    jx = ((xz[:, 0] - bounds[0]) / cell).astype(int)
    jz = ((xz[:, 1] - bounds[2]) / cell).astype(int)
    inside = (jx >= 0) & (jx < nx) & (jz >= 0) & (jz < nz)
    surf = _surface_height(jx[inside], jz[inside], y[inside], (nx, nz))

    on_surface = np.zeros(len(y), bool)
    on_surface[inside] = np.abs(y[inside] - surf[jx[inside], jz[inside]]) < band

    sel = np.zeros(len(y), bool)
    sel[inside] = road_mask[jx[inside], jz[inside]]
    sel &= on_surface
    road_xyz = np.column_stack([xz[sel, 0], y[sel], xz[sel, 1]])

    # one point per kerb cell, at the cell centre and the road's height there
    kx, kz = np.nonzero(kerb_mask)
    h = surf[kx, kz]
    good = np.isfinite(h)
    kerb_xyz = np.column_stack([bounds[0] + kx[good] * cell, h[good],
                                bounds[2] + kz[good] * cell])

    # Painted markings, given the road's height where each one sits.
    # Worth carrying separately from the kerb: kerbs and lane lines both
    # run ALONG the road and so only ever constrain sideways, but stop
    # bars, give-way lines and junction markings run ACROSS it, and those
    # are the only thing in the scene that can say how far along the road
    # a piece sits.
    px = ((paint_xz[:, 0] - bounds[0]) / cell).astype(int)
    pz = ((paint_xz[:, 1] - bounds[2]) / cell).astype(int)
    ok = (px >= 0) & (px < nx) & (pz >= 0) & (pz < nz)
    px, pz = px[ok], pz[ok]
    ph = surf[px, pz]
    good = np.isfinite(ph)
    paint_xyz = np.column_stack([bounds[0] + px[good] * cell, ph[good],
                                 bounds[2] + pz[good] * cell])
    return road_xyz, kerb_xyz, paint_xyz


def _surface_height(jx, jz, y, shape):
    """Height of the road surface per cell: a rough per-cell median,
    smoothed across neighbours so it tracks the road's slope instead of
    following whatever noise is in one cell."""
    nx, nz = shape
    flat = jx * nz + jz
    order = np.argsort(flat)
    f, yy = flat[order], y[order]
    idx, start = np.unique(f, return_index=True)
    ends = np.append(start[1:], len(f))
    rough = np.full(nx * nz, np.nan)
    for k, (s, e) in enumerate(zip(start, ends)):
        rough[idx[k]] = np.median(yy[s:e])
    rough = rough.reshape(nx, nz)
    filled = np.where(np.isnan(rough), np.nanmedian(rough), rough)
    return ndimage.median_filter(filled, size=9)
