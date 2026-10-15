"""Load the Street View round a typed place, its nearest pano the start."""
from ui.map_selection.load_area.area_page import area_page


def load_area(place, radius_input, *, find_nodes):
    """The map page for the area round place (lat, lon), as find_nodes finds it, with the
    radius still typed drawn round it."""
    found = find_nodes(*place)
    page = area_page(place, found, radius_input)
    return page
