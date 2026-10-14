"""Open a scene with one node per dot that has a downloaded candidate."""
from streetview_to_3d.common.Node import Node
from streetview_to_3d.common.Pano import Pano
from streetview_to_3d.common.Scene import Scene


def open_scene(prep):
    """A Scene with one node per dot that has a candidate (its best-ranked pano), from prep
    (panos.main.panos); the reconstruction fills in the rest."""
    best, elevations = {}, prep.get("elevations") or []
    for key, c in prep["catalog"].items():
        best.setdefault(c["dot"], (key, c))

    order = sorted(best)                       # dot index -> node index
    index = {dot: i for i, dot in enumerate(order)}
    nodes = []
    for dot in order:
        _, c = best[dot]
        nodes.append(Node(pano=Pano(
            source=c["source"], id=c["id"], lat=c["lat"], lon=c["lon"],
            date=c["date"], heading=c["heading"], pitch=c["pitch"], roll=c["roll"],
            elevation=elevations[dot] if dot < len(elevations) else c.get("elevation"))))

    adjacency = {str(index[d]): sorted(index[n] for n in ns if n in index)
                 for d, ns in prep["adjacency"].items() if d in index}
    return Scene(center=list(prep["center"]), nodes=nodes, adjacency=adjacency)
