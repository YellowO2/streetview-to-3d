"""Select every real Street View graph inside a circle, ready for Prepare."""
from ui.map_selection.expand_selection.selection_page import selection_page


def expand_selection(circle, *, expand):
    """The map page with everything expand walks inside circle selected, the circle drawn
    as an area whose edges drag."""
    found = expand(*circle)
    page = selection_page(circle, found)
    return page
