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
    """One panorama: where DA3 put it, where it really is, and how much of
    DA3's own reconstruction of it survived.

    views_kept out of views_total is DA3's solo score for this panorama --
    how many of its own views passed the consensus filter. Measured against
    real pairings (see the README): a score of 6 predicted 33% pairwise
    success, 13 and above predicted 100%. It is the one confidence number
    the reconstruction already knows about a single node.
    """
    key: str
    lat: float
    lon: float
    position: list[float]
    rotation: list[list[float]] | None = None
    date: str | None = None
    views_kept: int | None = None
    views_total: int | None = None

    @property
    def confidence(self):
        """Fraction of this panorama's views DA3 kept, or None if unrecorded."""
        if not self.views_total:
            return None
        return self.views_kept / self.views_total


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
                 position=list(v["position"]), rotation=v.get("rotation"),
                 date=v.get("date"), views_kept=v.get("n_views_kept"),
                 views_total=v.get("n_views_total"))
            for k, v in metadata.items()]
