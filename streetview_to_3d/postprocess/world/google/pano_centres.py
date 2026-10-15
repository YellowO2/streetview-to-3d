"""Where each placed pano's camera stands in the world."""
import numpy as np


def pano_centres(scene):
    """(n, 3) the world position (y down) of each of scene.placed()'s cameras."""
    return np.array([(np.asarray(nd.transform, float) @ [*nd.position, 1])[:3] for nd in scene.placed()])
