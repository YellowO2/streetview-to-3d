"""What a reconstructed area is, on disk and in memory.

One scene.json beside the clouds holds everything that is not geometry.

A NODE is a place we have a photograph of. It exists as soon as the street
graph is built, before any reconstruction: its Pano is what Street View or
Look Around told us about that spot. Reconstruction then fills in the rest
-- the node's own points, where DA3 put the camera, and once solved, where
that ply belongs in the world.

    node with no ply       we know where the street is, DA3 got nothing.
                           Still shapes the road lines and the ground.
    node with a ply        a full participant.

Points belong to nodes, never to groups of them: DA3 reconstructs one or
two panoramas at a time and a node's points enter the result exactly once,
so one panorama is the smallest thing ever independently produced.

A PIECE is not stored, because it is not a choice. Joining a node to a
piece composes a rigid transform into that piece's frame, so the nodes
sharing a DA3 frame are exactly the nodes joined by edges -- a connected
component of `edges`, and nothing else. Cutting an edge splits a piece;
bridging adds one and merges two. `Scene.pieces` derives them on demand.
"""
import json
import os
from dataclasses import asdict, dataclass, field

FILENAME = "scene.json"


@dataclass
class Pano:
    """What the source told us about one panorama. No reconstruction here.

    heading/pitch/roll are radians -- the source's own absolute measurement
    of which way the camera faced, unlike Node.rotation, which is DA3's and
    only means anything against the other nodes of the same piece.

    views_kept of views_total is DA3's solo score for this panorama: how
    many of its own views passed the consensus filter. Measured against
    real pairings (see the README), a score of 6 predicted 33% pairwise
    success and 13 predicted 100%.
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
        return f"{self.source}:{self.id}"

    @property
    def confidence(self):
        """Fraction of this panorama's views DA3 kept, or None if unknown."""
        if not self.views_total:
            return None
        return self.views_kept / self.views_total


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
    """A link DA3 reconstructed, between two nodes, by their index.

    keep_a/keep_b are (views kept, views total) for each end IN THE JOINT
    test -- how well the two panoramas agreed with each other, as opposed
    to how coherent either was alone.
    """
    a: int
    b: int
    keep_a: list[int] | None = None
    keep_b: list[int] | None = None

    @property
    def confidence(self):
        """The weaker end's keep rate -- a link is only as good as that."""
        rates = [k[0] / k[1] for k in (self.keep_a, self.keep_b) if k and k[1]]
        return min(rates) if rates else None


@dataclass
class Scene:
    center: list[float]
    nodes: list[Node] = field(default_factory=list)
    adjacency: dict[str, list[int]] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)

    @property
    def origin(self):
        return self.center[0], self.center[1]

    def pieces(self, min_confidence=None):
        """[[node index, ...], ...] -- the nodes sharing one DA3 frame.

        A connected component of the edges, so a piece is derived rather
        than stored. min_confidence ignores links DA3 was less sure of
        than that, which breaks a piece wherever its confidence ran out.
        """
        parent = list(range(len(self.nodes)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for e in self.edges:
            if min_confidence is not None and (e.confidence or 0) < min_confidence:
                continue
            a, b = find(e.a), find(e.b)
            if a != b:
                parent[a] = b

        groups = {}
        for i, n in enumerate(self.nodes):
            if n.ply is not None:
                groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    def neighbours(self, i):
        """{node index: Edge} -- who DA3 joined this node to."""
        return {(e.b if e.a == i else e.a): e
                for e in self.edges if i in (e.a, e.b)}

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
                   edges=[Edge(**e) for e in d["edges"]])
