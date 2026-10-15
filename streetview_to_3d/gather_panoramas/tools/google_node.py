"""Describe one Google pano as the node dict the map and candidates use."""
from streetview_to_3d.types.Pano import Pano


def google_node(pano_id, lat, lon, heading=None, **extra):
    """The node dict the map, the selection and the candidates all use for one Google pano."""
    return {"key": Pano.key_of("google", pano_id), "source": "google", "id": pano_id,
            "lat": lat, "lon": lon, "heading": heading, **extra}
