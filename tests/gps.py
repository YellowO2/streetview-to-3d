"""Reference-only: the old GPS-anchoring approach join_segments.py used to
use, removed from the real pipeline. Kept here for reference in case a
future need for true-north/real-world-oriented output comes back -- NOT
imported or used by anything in street_builder/ or services/.

Two mechanisms lived in street_builder/reconstruction/join_segments.py
that both got called "GPS logic", easy to conflate:

1. Bridge candidate pre-filter (BRIDGE_MAX_DIST_M / edge_max_dist_m):
   uses GPS distance only to narrow which node pairs are worth a real
   DA3 test. This one is NOT removed -- it's still in join_segments.py,
   since it's how two segments' metadata picks bridge candidates at all,
   not a placement mechanism.

2. This file: the final _fit_rigid_2d GPS-anchoring step that used to
   run inside join_segments() on every final piece (bridged or not) to
   place it in a shared real-world-meters frame. Removed because nothing
   downstream actually needs the result oriented to true north/real GPS
   coordinates -- only internally consistent, which bridging alone
   (mechanism 1) already provides for anything that successfully
   bridges. Pieces that never bridge now just stay in their own local
   DA3 frame and are returned separately (see join_segments.py's current
   join_segments()), rather than being force-placed via GPS.
"""
import numpy as np

from services.geo import latlon_to_local_m


def _fit_rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares rigid transform (2x2 rotation + 2D translation) mapping
    src onto dst, given Nx2 arrays matched row-by-row (N >= 2). Standard
    Kabsch algorithm restricted to 2D -- no scale, since DA3's own metric
    scale isn't being second-guessed here, only its horizontal position and
    heading against real GPS."""
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_c = src - src_mean
    dst_c = dst - dst_mean
    H = src_c.T @ dst_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, d])
    R = Vt.T @ D @ U.T
    t = dst_mean - R @ src_mean
    return R, t


def gps_anchor_segments(segments, node_entries):
    """The old join_segments() body, minus bridging. For each segment:
    fit rotation (about the vertical axis only -- GPS has no elevation
    data to fit against) + horizontal translation from that segment's
    own DA3-frame node positions onto their real GPS positions
    (converted to local meters, one shared origin for every segment).
    Vertical placement uses a simple heuristic instead -- shifting each
    segment's own average node height to a shared baseline.

    segments: list of (pts, cols, path_edges, date, reached,
    node_positions, frame_poses). node_entries: [(key, path, lat, lon,
    date), ...].

    Returns (points, colors, metadata): points/colors are one merged
    point cloud, all segments in the same real-world-meters frame.
    metadata is {key: {"lat", "lon", "date", "world_position": [x, y,
    z]}} for every node that contributed."""
    if not segments:
        raise ValueError("No segments to join.")

    by_key = {e[0]: e for e in node_entries}

    # Shared origin for every segment's lat/lon -> local-meters conversion --
    # arbitrary choice (first segment's first confirmed node), just needs
    # to be the SAME point for all of them so they land in one frame.
    first_key = next(iter(segments[0][5]))
    _, _, origin_lat, origin_lon, _ = by_key[first_key]

    all_pts, all_cols = [], []
    metadata = {}
    for seg_i, (pts, cols, path_edges, date, reached, node_positions, frame_poses) in enumerate(segments):
        keys = list(node_positions.keys())
        if len(keys) < 2:
            print(f"join: segment {seg_i} ({date}) has <2 confirmed nodes -- skipping, can't fit a rotation")
            continue

        da3_xz = np.array([[node_positions[k][0], node_positions[k][2]] for k in keys])
        real_en = []
        for k in keys:
            _, _, lat, lon, _ = by_key[k]
            e, n = latlon_to_local_m(lat, lon, origin_lat, origin_lon)
            real_en.append([e, n])
        real_en = np.array(real_en)

        R, t = _fit_rigid_2d(da3_xz, real_en)
        avg_y = float(np.mean([node_positions[k][1] for k in keys]))

        xz = pts[:, [0, 2]] @ R.T + t
        y = pts[:, 1] - avg_y
        transformed = np.column_stack([xz[:, 0], y, xz[:, 1]])

        heading = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
        print(f"join: segment {seg_i} ({date}) fit against {len(keys)} node(s), rotation={heading:.1f}deg")

        all_pts.append(transformed)
        all_cols.append(cols)

        node_xz = da3_xz @ R.T + t
        node_y = np.array([node_positions[k][1] for k in keys]) - avg_y
        for idx, k in enumerate(keys):
            _, _, lat, lon, node_date = by_key[k]
            metadata[k] = {
                "lat": lat, "lon": lon, "date": node_date,
                "world_position": [float(node_xz[idx, 0]), float(node_y[idx]), float(node_xz[idx, 1])],
            }

    if not all_pts:
        raise ValueError("No segment had enough confirmed nodes to fit.")

    return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0), metadata
