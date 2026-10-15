"""Give each node its share of the one ground, and its trees their sway."""
import numpy as np

from streetview_to_3d.types.Cloud import Cloud

SWAY_M = 6.0      # vegetation sways fully at this height over the ground


def handed_out(scene, clouds, ground, scene_ground, cameras):
    """The scene with each placed node's cloud (clouds[k], in the world) back in its own
    frame, plus the ground its pano painted, and a sway per point; returns the scene."""
    for k, n in enumerate(scene.placed()):
        mine = ground.painter == k
        x = np.concatenate([clouds[k].points, ground.points[mine]])
        # vegetation sways more the higher it is over the ground (world: y down)
        tree = cameras[k].classed(clouds[k].points, ("vegetation",), False)
        over = -clouds[k].points[:, 1] - scene_ground.at(clouds[k].points[:, [0, 2]])[1]
        sway = np.r_[tree * np.clip(np.nan_to_num(over) / SWAY_M, 0, 1), np.zeros(int(mine.sum()))]
        T = np.asarray(n.transform, float)
        n.cloud = Cloud((x - T[:3, 3]) @ np.linalg.inv(T[:3, :3]).T,
                        np.concatenate([clouds[k].colours, ground.colours[mine]]), sway)
    return scene
