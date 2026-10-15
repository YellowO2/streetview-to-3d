"""Load each placed node's pano photo and its class map."""
import glob
import os

import numpy as np
from PIL import Image

from streetview_to_3d.common.streetview_fetch.fetch_da3_pano import fetch_da3_pano
from streetview_to_3d.models.pano_labels import pano_labels


def photos(scene):
    """The scene with each placed node's photo set: its Google pano's (image path, class map),
    else None. The class map is the newest the walk made (models/pano_labels.py), else made now."""
    for n in scene.placed():
        n.photo = _photo(n.pano)
    return scene


def _photo(pano):
    if pano.source != "google":
        return None
    path = fetch_da3_pano(pano.id)
    if not path:
        return None
    made = sorted(glob.glob(f"{glob.escape(path)}.*.labels.png"), key=os.path.getmtime)
    return path, (np.asarray(Image.open(made[-1])) if made else pano_labels(path, device="cpu"))
