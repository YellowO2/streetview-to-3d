"""A link DA3 reconstructed between two nodes."""
from dataclasses import dataclass


@dataclass
class Edge:
    """A link DA3 reconstructed between two nodes (by index); keep_a/keep_b are
    each end's (views kept, views total) in the joint run."""
    a: int
    b: int
    keep_a: list[int] | None = None
    keep_b: list[int] | None = None
