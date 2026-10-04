"""What a reconstructed area is, on disk (one scene.json beside the clouds) and in memory.

A node is a place we have a photograph of; its points come from one panorama.
A piece is the nodes sharing one DA3 frame: a connected component of `edges`, derived, never stored.
"""
import json
import os
from dataclasses import asdict, dataclass, field


FILENAME = "scene.json"


def node_key(source, pano_id):
    return f"{source}:{pano_id}"


class DisjointSet:
    """Union-find over items 0..n-1."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


@dataclass
class Pano:
    """What the source told us about one panorama.

    heading/pitch/roll are the source's absolute radians (Node.rotation is DA3's, per piece);
    views_kept of views_total is how many of its views passed DA3's consensus filter.
    """
    source: str
    id: str
    lat: float
    lon: float
    date: str | None = None
    elevation: float | None = None
    heading: float | None = None
    pitch: float | None = None
    roll: float | None = None
    views_kept: int | None = None
    views_total: int | None = None

    @property
    def key(self):
        return node_key(self.source, self.id)


@dataclass
class Node:
    """One place, its photograph, and whatever DA3 made of it."""
    pano: Pano
    ply: str | None = None
    position: list[float] | None = None
    rotation: list[list[float]] | None = None
    transform: list[list[float]] | None = None

    @property
    def key(self):
        return self.pano.key


@dataclass
class Edge:
    """A link DA3 reconstructed between two nodes (by index); keep_a/keep_b are
    each end's (views kept, views total) in the joint run."""
    a: int
    b: int
    keep_a: list[int] | None = None
    keep_b: list[int] | None = None


@dataclass
class Scene:
    center: list[float]
    nodes: list[Node] = field(default_factory=list)
    adjacency: dict[str, list[int]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    # the land around the scene, a triangle .ply beside scene.json already
    # in the world, and its roads and bridges, points (postprocess/world/terrain.py)
    land: str | None = None
    terrain: str | None = None
    # its roads further out, a triangle .ply (the near ones are points in terrain)
    roads: str | None = None
    # the OSM buildings around it, spaced finer than the land (same module)
    buildings: str | None = None
    # its water, flat outlines each at a level (postprocess/world/water.py)
    water: str | None = None
    # the far buildings, solid: a triangle .ply (postprocess/world/buildings.py's solid)
    blocks: str | None = None
    # what moves round it -- its cars' roads, its birds, its boats' courses (postprocess/world/life.py)
    life: str | None = None
    # experimental: Google 3D Tiles points standing in for the land, roads and buildings
    # (postprocess/world/google.py)
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
        with open(path, "w") as f:
            json.dump(asdict(self), f)
        return path

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
