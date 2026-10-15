"""Show the selection after the area is dragged, all of it selected."""
from ui.map_selection.tools.map_view import map_page


def dragged_page(drag, state, found):
    """The map page for the nodes and links found inside the dragged area, all selected,
    reopened where the map was when the drag ended."""
    nodes, edges = found
    view = drag.view
    return map_page({**state, "nodes": nodes, "edges": edges, "selected": [n["key"] for n in nodes],
                     "selected_edges": list(edges), "area": drag.area, "radius_m": None,
                     "view": (view["lat"], view["lon"], view["zoom"]) if view else state.get("view")})
