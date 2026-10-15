"""A selected street corridor, its panoramas downloaded."""
from dataclasses import dataclass


@dataclass
class Corridor:
    # Per capture date, best first: {"date", "dot_candidates": {dot: [(key, path, lat, lon)]}}.
    date_graphs: list
    # The corridor's dots, (lat, lon), and which dots link to which: {dot: [dot]}.
    points: list
    adjacency: dict
    # Ground height at each dot, in metres.
    elevations: list
    # Every downloaded pano by key: {dot, source, id, lat, lon, date, heading, pitch, roll}.
    catalog: dict
    # Where the walk starts and the scene's centre, (lat, lon).
    start: tuple
    center: tuple
