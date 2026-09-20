"""What a reconstructed area is, on disk and in memory.

One scene.json beside the clouds holds everything that is not geometry:

    center     the coordinate that was searched for. Every position in
               postprocess is metres from it, and saved transforms are
               relative to it, so it is written once and never recomputed.
    graph      the street graph the area was reconstructed from. Road
               alignment measures against lines built from it, so an area
               carries its own rather than borrowing one dataset's.
    pieces     each cloud and, for every panorama that built it, where DA3
               put the camera beside where the camera really was. That
               pairing is what makes the GPS fit possible.

A piece is identified by its position in the list; nothing carries an id,
because splitting renumbers pieces and no id survives it.
"""
import json
import os
from dataclasses import asdict, dataclass, field

FILENAME = "scene.json"


@dataclass
class Node:
    """One panorama: where DA3 put it, and where it really is."""
    key: str
    lat: float
    lon: float
    position: list[float]
    date: str | None = None


@dataclass
class Piece:
    """One cloud that moves as a single rigid body."""
    ply: str
    nodes: list[Node]

    def __len__(self):
        return len(self.nodes)


@dataclass
class Graph:
    """Street View's dots and how they link."""
    points: list[list[float]]
    adjacency: dict[str, list[int]]


    @classmethod
    def read(cls, path):
        """From a standalone graph file, which may carry more than this."""
        with open(path) as f:
            d = json.load(f)
        return cls(points=d["points"], adjacency=d["adjacency"])


@dataclass
class Scene:
    center: list[float]
    graph: Graph
    pieces: list[Piece] = field(default_factory=list)

    @property
    def origin(self):
        return self.center[0], self.center[1]

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
                f"{path} is missing -- it holds the origin, the street graph "
                "and every camera's real position, so the clouds beside it "
                "cannot be placed without it.")
        with open(path) as f:
            d = json.load(f)
        return cls(center=d["center"],
                   graph=Graph(**d["graph"]),
                   pieces=[Piece(ply=p["ply"],
                                 nodes=[Node(**n) for n in p["nodes"]])
                           for p in d["pieces"]])


def from_metadata(metadata):
    """[Node] from street_builder's per-panorama reconstruction metadata."""
    return [Node(key=k, lat=v["lat"], lon=v["lon"],
                 position=list(v["position"]), date=v.get("date"))
            for k, v in metadata.items()]
