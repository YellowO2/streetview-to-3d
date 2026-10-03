"""The walk's pieces in the shape the scene is saved from; each is placed by its GPS later."""


def _links_by_node(path_edges):
    """{node key: {neighbour key: [views kept, views total]}}."""
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
    """[(clouds, metadata), ...], one per piece: clouds is {node key: (points, colors)};
    metadata per node is its lat/lon/date, pose in the piece's frame, view counts and links."""
    results = []
    for clouds, path_edges, date, frame_poses in pieces:
        links = _links_by_node(path_edges)
        metadata = {k: {"lat": lat, "lon": lon, "date": date, "position": pos.tolist(),
                        "rotation": rot.tolist(), "n_views_kept": n_kept,
                        "n_views_total": n_total,
                        **({"links": links[k]} if k in links else {})}
                    for k, (pos, rot, path, lat, lon, n_kept, n_total) in frame_poses.items()}
        results.append(({k: v for k, v in clouds.items() if k in metadata}, metadata))
    return results
