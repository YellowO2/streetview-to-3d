"""The buildings stood round the scene, as points and solid meshes."""
from dataclasses import dataclass

from streetview_to_3d.build_scene.postprocess.world.Outlines import Outlines
from streetview_to_3d.build_scene.postprocess.world.buildings.Blocks import Blocks


@dataclass
class Buildings:
    """outlines: what they stand on (Outlines); blocks: the points of those DA3 reaches (Blocks);
    solid: blocks.ply's mesh of the rest (write_mesh's arguments), None without any."""
    outlines: Outlines
    blocks: Blocks
    solid: dict | None
