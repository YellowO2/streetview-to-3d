"""Load each placed node's pano photo and its class map."""
import glob
import os

import numpy as np
from PIL import Image

from streetview_to_3d.models.Segmenter import Segmenter
from streetview_to_3d.postprocess.tools.fetch_da3_pano import fetch_da3_pano


def photos(scene, labeller: Segmenter):
    """The scene with each placed node's photo set: its Google pano's (image path, class map),
    else None. The class map is the newest the walk made (Segmenter.pano_labels), else
    labeller makes it now."""
    for n in scene.placed():
        n.photo = _photo(n.pano, labeller)
    return scene


def _photo(pano, labeller):
    if pano.source != "google":
        return None
    path = fetch_da3_pano(pano.id)
    if not path:
        return None
    made = sorted(glob.glob(f"{glob.escape(path)}.*.labels.png"), key=os.path.getmtime)
    return path, (np.asarray(Image.open(made[-1])) if made else labeller.pano_labels(path))
