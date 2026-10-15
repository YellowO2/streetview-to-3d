"""Set each placed node's pano photo: its image beside its class map."""
from streetview_to_3d.build.postprocess.tools.fetch_da3_pano import fetch_da3_pano


def photos(scene):
    """The scene with each placed node's photo set: its Google pano's (image path, class map),
    the class map the one the walk made (Node.labels), else None."""
    for n in scene.placed():
        n.photo = _photo(n.pano, n.labels)
    return scene


def _photo(pano, labels):
    if pano.source != "google" or labels is None:
        return None
    path = fetch_da3_pano(pano.id)
    return (path, labels) if path else None
