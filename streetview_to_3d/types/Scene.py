"""The scene: its nodes, pieces and edges, on disk and in memory.

What a reconstructed area is, on disk (one scene.json beside the clouds) and in memory.

A node is a place we have a photograph of; its points come from one panorama.
A piece is the nodes sharing one DA3 frame: a connected component of `edges`, derived, never stored.
"""
import json
import os
from dataclasses import asdict, dataclass, field, replace

from streetview_to_3d.tools.DisjointSet import DisjointSet
from streetview_to_3d.types.Edge import Edge
from streetview_to_3d.types.Node import Node
from streetview_to_3d.types.Pano import Pano


FILENAME = "scene.json"
# beside it, each placed node's class map as <pano id>.png (save.py, load.py)
LABELS = "labels"


@dataclass
class Scene:
    center: list[float]
    nodes: list[Node] = field(default_factory=list)
    adjacency: dict[str, list[int]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    # the land around the scene, a triangle .ply beside scene.json already
    # in the world, and its roads and bridges, points (postprocess/world/land/, roads/)
    land: str | None = None
    terrain: str | None = None
    # its roads further out, a triangle .ply (the near ones are points in terrain)
    roads: str | None = None
    # the OSM buildings around it, spaced finer than the land (same module)
    buildings: str | None = None
    # its water, flat outlines each at a level (postprocess/world/Water.py)
    water: str | None = None
    # the far buildings, solid: a triangle .ply (postprocess/world/buildings/solid_blocks.py)
    blocks: str | None = None
    # what moves round it -- its cars' roads, its birds, its boats' courses (postprocess/world/life.py, car_roads.py)
    life: str | None = None
    # experimental: Google 3D Tiles points standing in for the land, roads and buildings
    # (postprocess/world/google/)
    google: str | None = None

    @property
    def origin(self):
        return self.center[0], self.center[1]

    def pieces(self):
        """[[node index, ...], ...]: the nodes sharing one DA3 frame."""
        sets = DisjointSet(len(self.nodes))
        for e in self.edges:
            sets.union(e.a, e.b)
        groups = {}
        for i, n in enumerate(self.nodes):
            # membership is having a DA3 pose, not points
            if n.position is not None:
                groups.setdefault(sets.find(i), []).append(i)
        return list(groups.values())

    def save(self, directory):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, FILENAME)
        d = asdict(replace(self, nodes=[replace(n, cloud=None, labels=None, photo=None) for n in self.nodes]))
        for n in d["nodes"]:
            del n["cloud"], n["labels"], n["photo"]      # saved beside it
        with open(path, "w") as f:
            json.dump(d, f)
        return path

    def placed(self):
        """The nodes with points and a place in the world."""
        return [n for n in self.nodes if n.ply and n.transform]

    @classmethod
    def load(cls, directory):
        path = os.path.join(directory, FILENAME)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is missing -- it holds the origin, every place we "
                "have a photograph of, and where its points belong, so the "
                "clouds beside it cannot be placed without it.")
        with open(path) as f:
            d = json.load(f)
        return cls(center=d["center"],
                   nodes=[Node(pano=Pano(**n.pop("pano")), **n) for n in d["nodes"]],
                   adjacency={str(k): v for k, v in d["adjacency"].items()},
                   edges=[Edge(**e) for e in d["edges"]],
                   land=d.get("land"), terrain=d.get("terrain"), roads=d.get("roads"), buildings=d.get("buildings"), water=d.get("water"),
                   blocks=d.get("blocks"), life=d.get("life"), google=d.get("google"))

