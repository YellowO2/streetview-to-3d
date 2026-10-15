"""The scene's panos as cameras, with their photos' masks read from disk."""
import logging

from streetview_to_3d.postprocess.world.PanoViews import PanoViews

logger = logging.getLogger(__name__)


def pano_views(scene):
    """PanoViews of the scene's placed panos, each photo with its drop mask (the rig's blur and
    what pano_mask drops); none if their photos cannot be read."""
    from streetview_to_3d.models.pano_mask import pano_mask
    from streetview_to_3d.postprocess.Camera import Camera
    from streetview_to_3d.postprocess.tools.paint import blurred
    try:
        nodes = [nd for nd in scene.placed() if nd.rotation is not None]
        photos = []
        for nd in nodes:
            ph = nd.photo
            photos.append(ph and (ph[0], ph[1], pano_mask(ph[1]) | blurred(ph[0])))
        return PanoViews([Camera(nd) for nd in nodes], photos)
    except (OSError, ValueError) as e:
        logger.warning(f"terrain: no pano photos ({e!r})")
        return PanoViews()
