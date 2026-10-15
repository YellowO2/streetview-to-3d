"""Select every real Street View graph inside a circle, ready for Prepare."""
from ui.map_selection.expand_selection.selection_page import selection_page


def expand_selection(circle, *, expand_area):
    """The map page with everything expand_area walks inside circle selected, the circle drawn
    as an area whose edges drag."""
    found = expand_area(*circle)
    page = selection_page(circle, found)
    return page
