"""What a reconstructed area is, on disk and in memory.

One scene.json beside the clouds holds everything that is not geometry:

    center     the coordinate that was searched for. Every position in
               postprocess is metres from it, and saved transforms are
               relative to it, so it is written once and never recomputed.
    graph      the street graph the area was reconstructed from. Road
               alignment measures against lines built from it, so an area
               carries its own rather than borrowing one dataset's.
    pieces     which nodes DA3 managed to reconstruct together, and how
               confident it was about each link

Points belong to NODES, not to pieces: DA3 only ever reconstructs one or
two panoramas at a time, and a node's own points enter the result exactly
once, so the smallest thing that was ever independently reconstructed is
one panorama. A piece owns no bytes -- it is a grouping of nodes, and
regrouping them costs nothing.

A piece is identified by its position in the list; nothing carries an id,
because regrouping renumbers pieces and no id survives it.
"""
import json
import os
from dataclasses import asdict, dataclass, field

FILENAME = "scene.json"


@dataclass
class Node:
    """One panorama: its own points, where DA3 put it, where it really is,
    and how much of DA3's own reconstruction of it survived.

    views_kept out of views_total is DA3's solo score for this panorama --
    how many of its own views passed the consensus filter. Measured against
    real pairings (see the README): a score of 6 predicted 33% pairwise
    success, 13 and above predicted 100%. It is the one confidence number
    the reconstruction already knows about a single node.

    heading/pitch/roll are radians, and are the source's own measurement of
    which way the camera faced -- absolute, unlike `rotation`, which is
    DA3's and is only meaningful against the other nodes of the same piece.
    """
    key: str
    lat: float
    lon: float
    position: list[float]
    ply: str | None = None
    rotation: list[list[float]] | None = None
    date: str | None = None
    views_kept: int | None = None
    views_total: int | None = None
    heading: float | None = None
    pitch: float | None = None
    roll: float | None = None

    @property
    def confidence(self):
        """Fraction of this panorama's views DA3 kept, or None if unrecorded."""
        if not self.views_total:
            return None
        return self.views_kept / self.views_total


@dataclass
class Edge:
    """A link DA3 actually reconstructed, between two nodes of one piece.

    keep_a/keep_b are (views kept, views total) for each end IN THE JOINT
    test -- how well the two panoramas agreed with each other, as opposed
    to how coherent either was alone. A piece is exactly a run of nodes
    joined by edges like these, so where the edges stop is where DA3 ran
    out of confidence.
    """
    a: str
    b: str
    keep_a: list[int] | None = None
    keep_b: list[int] | None = None

    @property
    def confidence(self):
        """The weaker end's keep rate -- a link is only as good as that."""
        rates = [k[0] / k[1] for k in (self.keep_a, self.keep_b) if k and k[1]]
        return min(rates) if rates else None


@dataclass
class Piece:
    """The nodes that move as one rigid body.

    A graph, not a bag: `edges` are the links DA3 confirmed. Edges live
    here rather than on each Node because the relation is symmetric, and
    stored once it cannot disagree with itself.
    """
    nodes: list[Node]
    edges: list[Edge] = field(default_factory=list)

    def __len__(self):
        return len(self.nodes)

    def neighbours(self, key):
        """{neighbour key: Edge} for one node."""
        return {(e.b if e.a == key else e.a): e
                for e in self.edges if key in (e.a, e.b)}


@dataclass
class Graph:
    """Street View's dots, how they link, and how high the ground is.

    elevations are metres above sea level per dot, straight from the pano
    lookup that found each one -- Y-UP, unlike everything downstream.
    """
    points: list[list[float]]
    adjacency: dict[str, list[int]]
    elevations: list[float | None] | None = None

    @classmethod
    def read(cls, path):
        """From a standalone graph file, which may carry more than this."""
        with open(path) as f:
            d = json.load(f)
        return cls(points=d["points"], adjacency=d["adjacency"],
                   elevations=d.get("elevations"))


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
                   pieces=[Piece(nodes=[Node(**n) for n in p["nodes"]],
                                 edges=[Edge(**e) for e in p["edges"]])
                           for p in d["pieces"]])


def from_metadata(metadata):
    """([Node], [Edge]) from the reconstruction's per-panorama metadata.

    Each node's metadata lists its own links, so an edge appears twice --
    once from each end. They are collapsed to one Edge here.
    """
    nodes = [Node(key=k, lat=v["lat"], lon=v["lon"],
                  position=list(v["position"]), rotation=v.get("rotation"),
                  date=v.get("date"), views_kept=v.get("n_views_kept"),
                  views_total=v.get("n_views_total"))
             for k, v in metadata.items()]

    edges, seen = [], set()
    for a, v in metadata.items():
        for b, keep in (v.get("links") or {}).items():
            if b not in metadata or (b, a) in seen:
                continue
            seen.add((a, b))
            edges.append(Edge(a=a, b=b, keep_a=keep,
                              keep_b=(metadata[b].get("links") or {}).get(a)))
    return nodes, edges
