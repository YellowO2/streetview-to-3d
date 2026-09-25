"""The panoramas around one clicked node, for work centred on a single
pano (a splat): which captures exist at the node, and for each, the same
capture's nearest pano at every node it links to.

Same capture means same source AND same date. Panos of different captures
are never DA3-linked (see walk_graph), and a Google and an Apple pano of
the same month are still two different captures.
"""
from streetview_to_3d.build_street_graph.build_graph import cap_bucket_for_date
from streetview_to_3d.build_street_graph.date_ranking import date_recency_key
from streetview_to_3d.build_street_graph.fetch_nodes import fetch_corridor_nodes
from streetview_to_3d.services.geo import haversine_m


def panos_around(node, neighbours):
    """node: (lat, lon) of the clicked node; neighbours: the (lat, lon) of
    each node it links to on the map.

    Returns [{"target": pano, "neighbours": [pano, ...]}, ...], one per
    capture at the node, newest first. A pano is fetch_nodes' candidate
    dict (key, source, id, lat, lon, date, heading, ...). A capture with no
    pano at any neighbour still comes back, with no neighbours: DA3 then
    has the target alone.
    """
    if not neighbours:
        raise ValueError("this node links to nothing on the map")
    buckets, points, adjacency, _ = fetch_corridor_nodes([(node, n) for n in neighbours])
    t = min(range(len(points)), key=lambda i: haversine_m(*node, *points[i]))

    captures = {}
    for p in sorted(buckets[t], key=lambda p: haversine_m(*node, p["lat"], p["lon"])):
        captures.setdefault((p["source"], p["date"]), p)

    out = []
    for (source, date), target in sorted(captures.items(),
                                         key=lambda c: date_recency_key(c[0][1]), reverse=True):
        near = []
        for i in adjacency.get(t, []):
            same_source = [p for p in buckets[i] if p["source"] == source]
            near += cap_bucket_for_date(same_source, date, *points[i], 1)
        out.append({"target": target, "neighbours": near})
    return out
