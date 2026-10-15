"""DA3's points of a placed scene in the world, thinned."""
import numpy as np

SCENE_EVERY = 4                            # scene points subsampled to every this many-th


def scene_points(sc, every=SCENE_EVERY):
    """(points, colours) of every placed node's cloud in the world, every every-th point."""
    pts, cols = [np.zeros((0, 3))], [np.zeros((0, 3))]
    for nd in sc.placed():
        pts.append(nd.cloud.world(nd.transform, every))
        cols.append(nd.cloud.colours[::every])
    return np.concatenate(pts), np.concatenate(cols)
