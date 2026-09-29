"""The walk's pieces in the shape the scene is saved from.

A piece is the nodes DA3 linked into one frame. Pieces the walk left
separate stay separate: each is placed by its own nodes' GPS later, in
postprocess/.
"""


def _links_by_node(path_edges):
    """{node key: {neighbour key: [views kept, views total]}}.

    Stored per node so it survives the per-node metadata JSON the Hub
    already holds.
    """
    links = {}
    for a, b, keep_a, keep_b in path_edges:
        links.setdefault(a, {})[b] = keep_a
        links.setdefault(b, {})[a] = keep_b
    return links


def _piece_edges(metadata):
    """[(a, b, keep_a, keep_b)] rebuilt from the links in `metadata`."""
    edges, seen = [], set()
    for a, m in metadata.items():
        for b, keep_a in (m.get("links") or {}).items():
            if b not in metadata or (b, a) in seen:
                continue
            seen.add((a, b))
            edges.append((a, b, keep_a, (metadata[b].get("links") or {}).get(a)))
    return edges


def pieces_to_output(pieces):
    """[(clouds, metadata), ...] -- one entry per still-separate piece.

    clouds is {node key: (points, colors)}: DA3 only ever reconstructs one
    or two panoramas at a time and a node's points enter exactly once, so
    every point belongs to a known node and nothing needs re-deriving it.

    metadata carries, per node, its real lat/lon/date, its position and
    rotation in this piece's frame, the view counts behind DA3's own
    confidence in it, and its links to the nodes it was reconstructed with.
    """
    results = []
    for clouds, path_edges, date, reached, node_positions, frame_poses in pieces:
        links = _links_by_node(path_edges)
        metadata = {k: {"lat": lat, "lon": lon, "date": date, "position": pos.tolist(),
                        "rotation": rot.tolist(), "n_views_kept": n_kept,
                        "n_views_total": n_total,
                        **({"links": links[k]} if k in links else {})}
                    for k, (pos, rot, path, lat, lon, n_kept, n_total) in frame_poses.items()}
        results.append(({k: v for k, v in clouds.items() if k in metadata}, metadata))
    return results
