"""Choose the corridor's panoramas for each capture date and download them."""
from streetview_to_3d.panos.date_graphs import date_graphs
from streetview_to_3d.panos.downloaded import downloaded


def panos(start, goals, corridor_edges, center):
    """What the GPU step takes (see downloaded.py), with no GPU: kept apart so the GPU click
    is a fresh interaction, as the ZeroGPU token expires on wall-clock time. start/center:
    (lat, lon); goals: [(lat, lon)], the other selected nodes; corridor_edges: the
    selection's real Street View links, [((lat, lon, id), (lat, lon, id))]."""
    graphs = date_graphs(corridor_edges, start, goals)
    corridor = downloaded(graphs, start, center)
    return corridor
