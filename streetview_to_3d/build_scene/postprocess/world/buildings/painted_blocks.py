"""Building points coloured from the panos and met to DA3's."""
from streetview_to_3d.build_scene.postprocess.world.buildings.Blocks import ROOF
from streetview_to_3d.build_scene.postprocess.world.tools.pano_paint import pano_paint
from streetview_to_3d.build_scene.postprocess.world.tools.seams import toward


def painted_blocks(blocks, site, views):
    """blocks (Blocks) painted from the panos (views: PanoViews) near the cameras but on roofs,
    and turning into DA3's points (site: Site) near them; painted: how many, near: by how much."""
    alone = ~blocks.reached[blocks.which]                          # no DA3 near: not pano-painted
    blocks.cols, blocks.painted = pano_paint(views, blocks.pts, blocks.cols, site.points,
                                             skip=(blocks.edge == ROOF) | alone)
    if len(blocks.pts):
        blocks.cols, blocks.near = toward(blocks.pts, blocks.cols, site.tree, site.colours)
    return blocks
