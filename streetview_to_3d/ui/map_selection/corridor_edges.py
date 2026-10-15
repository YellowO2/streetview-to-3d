"""List the map selection's links with each end's place and pano."""
from streetview_to_3d.ui.map_selection.nodes_by_key import nodes_by_key


def corridor_edges(state):
    """The selected edges as ((lat, lon, pano_id), (lat, lon, pano_id))
    pairs -- what common/corridor_points.py builds the spots from."""
    by_key = nodes_by_key(state)
    node = lambda k: (by_key[k]["lat"], by_key[k]["lon"], by_key[k]["id"])
    return [(node(a), node(b)) for a, b in state.get("selected_edges", []) if a in by_key and b in by_key]
