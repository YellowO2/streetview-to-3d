"""Combine independently-reconstructed pathfind segments into one merged
point cloud.

Each segment (see run_pathfind_reconstruction) is placed by DA3 in its own
arbitrary frame, unrelated to any other segment's -- there's no assumption
segments overlap or share any node at all (a dead-end restart picks
whatever date works best from the dead-end's position, which is often a
different date than the segment before it). So instead of matching
segments against each other, each one is independently fit against real
GPS using only its own confirmed nodes, then all of them land in the same
shared real-world-meters frame and can just be concatenated.

No GPU needed -- this is plain linear algebra (2D Kabsch/Procrustes), runs
client-side.
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


def join_segments(segments, node_entries) -> tuple[np.ndarray, np.ndarray]:
    """segments: run_prepared_pathfind_segments' output -- list of (pts,
    cols, path_edges, date, reached, node_positions), node_positions being
    {key: np.ndarray(3,)} for that segment's own confirmed nodes.
    node_entries: prep['node_entries'] -- (key, path, lat, lon, date)
    tuples, giving real lat/lon for every node key referenced anywhere.

    For each segment: fit rotation (about the vertical axis only -- GPS
    has no elevation data to fit against, so the vertical axis is left
    alone rather than risk an unconstrained tilt) + horizontal translation
    from that segment's own DA3-frame node positions onto their real GPS
    positions (converted to local meters, one shared origin for every
    segment). Vertical placement uses a simple heuristic instead --
    shifting each segment's own average node height to a shared baseline,
    since GPS can't fit that axis and street-level camera heights should
    be roughly comparable across segments anyway.

    Returns (points, colors) -- one merged point cloud, all segments in
    the same real-world-meters frame.
    """
    if not segments:
        raise ValueError("No segments to join.")

    by_key = {e[0]: e for e in node_entries}

    # Shared origin for every segment's lat/lon -> local-meters conversion --
    # arbitrary choice (first segment's first confirmed node), just needs
    # to be the SAME point for all of them so they land in one frame.
    first_key = next(iter(segments[0][5]))
    _, _, origin_lat, origin_lon, _ = by_key[first_key]

    all_pts, all_cols = [], []
    for seg_i, (pts, cols, path_edges, date, reached, node_positions) in enumerate(segments):
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

    if not all_pts:
        raise ValueError("No segment had enough confirmed nodes to fit.")

    return np.concatenate(all_pts, axis=0), np.concatenate(all_cols, axis=0)
