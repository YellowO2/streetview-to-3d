"""Put the walk's pieces into the scene: each node's pano, pose and points."""
from streetview_to_3d.common.Edge import Edge
from streetview_to_3d.common.Pano import Pano
from streetview_to_3d.reconstruct.thinned import thinned


def assemble(sc, pieces, prep):
    """The scene (open_scene) with each piece's nodes filled in: the pano that filled it, DA3's
    pose, its points thinned (thinned.py) and named node_<i>.ply, and the edges by node index.
    prep: what panos.main.panos prepared; its catalog matches each pano to its dot.

    Patches overlap, so two pieces can hold the same place: the bigger keeps it and the
    other's pano there is dropped with its links (keeping both glued two frames together).
    """
    catalog = prep["catalog"]
    node_of_dot = {catalog[n.key]["dot"]: i
                   for i, n in enumerate(sc.nodes) if n.key in catalog}

    taken = set()
    for p_i in sorted(range(len(pieces)), key=lambda k: -len(pieces[k][1])):
        clouds, metadata = pieces[p_i]
        placed = {}
        for key, m in metadata.items():
            c = catalog.get(key)
            if c is None or c["dot"] not in node_of_dot:
                continue
            i = node_of_dot[c["dot"]]
            if i in taken:
                print(f"save: piece {p_i}'s {key} is at node {i}, already held by a "
                      f"bigger piece -- dropped with its links", flush=True)
                continue
            taken.add(i)
            node = sc.nodes[i]
            node.pano = Pano(
                source=c["source"], id=c["id"], lat=m["lat"], lon=m["lon"],
                date=m.get("date"), elevation=node.pano.elevation,
                heading=c["heading"], pitch=c["pitch"], roll=c["roll"],
                views_kept=m.get("n_views_kept"), views_total=m.get("n_views_total"))
            node.position = list(m["position"])
            node.rotation = m.get("rotation")
            # no points: the camera still places the piece, but an empty .ply breaks the viewer
            pts, cols = clouds[key]
            if len(pts):
                node.ply = f"node_{i}.ply"
                node.cloud = thinned(pts, cols)
            placed[key] = i

        for a, b, keep_a, keep_b in _piece_edges(metadata):
            if a in placed and b in placed:
                sc.edges.append(Edge(a=placed[a], b=placed[b],
                                               keep_a=keep_a, keep_b=keep_b))
    return sc


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


