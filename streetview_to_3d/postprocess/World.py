"""The world laid around a scene, as the files that will hold it."""
from dataclasses import dataclass, field


@dataclass
class World:
    """Each file's contents by its name in the scene: meshes and points as ply_io.write_mesh
    and write_ply's arguments, the rest as JSON."""
    meshes: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)
    json: dict = field(default_factory=dict)
