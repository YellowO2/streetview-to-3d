"""Select everything inside the area as its edges are dragged."""
from ui.map_selection.redraw_area.dragged_page import dragged_page


def redraw_area(drag, state, *, expand_area):
    """The map page with everything expand_area walks inside the dragged area selected: from
    what it has already looked up, so pulling an edge in fetches nothing and pushing it
    out only what is new."""
    found = expand_area(*drag.centre, area=drag.area)
    page = dragged_page(drag, state, found)
    return page
