"""Label a whole pano once, and keep its class map beside it on disk."""
import os

import numpy as np
from PIL import Image

from streetview_to_3d.models.segment import MODEL_ID, segment


def pano_labels(path, model_id=None, device=None, saved=None):
    """A whole pano's class map (ids as in segment.CLASSES), kept in saved (default
    beside it, named by the model) for whatever needs it later -- the scene
    carries a copy (save.py). A saved one is read, not redone: the walk
    runs DA3 on one pano many times. model_id, device: see segment."""
    saved = saved or f"{path}.{(model_id or MODEL_ID).split('/')[-1]}+water.labels.png"
    if os.path.exists(saved):
        return np.asarray(Image.open(saved))
    labels = segment([path], model_id, device)[0][0]
    os.makedirs(os.path.dirname(saved) or ".", exist_ok=True)
    Image.fromarray(labels).save(saved)
    return labels
