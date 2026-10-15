"""Select everything inside the area as its edges are dragged."""
from ui.map_selection.redraw_area.dragged_page import dragged_page


def redraw_area(drag, state, *, expand):
    """The map page with everything expand walks inside the dragged area selected: from
    what it has already looked up, so pulling an edge in fetches nothing and pushing it
    out only what is new."""
    found = expand(*drag.centre, area=drag.area)
    page = dragged_page(drag, state, found)
    return page
