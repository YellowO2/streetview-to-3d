"""Read the map's state: its loaded nodes, and the links selected on it."""


def nodes_by_key(state):
    """{key: node dict} of every node the map state has loaded."""
    return {n["key"]: n for n in state["nodes"]}


def corridor_edges(state):
    """The selected edges as ((lat, lon, pano_id), (lat, lon, pano_id))
    pairs -- what tools/corridor_points.py builds the spots from."""
    by_key = nodes_by_key(state)
    node = lambda k: (by_key[k]["lat"], by_key[k]["lon"], by_key[k]["id"])
    return [(node(a), node(b)) for a, b in state.get("selected_edges", []) if a in by_key and b in by_key]
