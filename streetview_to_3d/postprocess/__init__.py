"""Placing what was reconstructed: separate clouds into one scene. No GPU.

    pipeline.py   the whole thing as one call: place, then fill
    place.py      every piece by its panoramas' GPS, elevation and orientation
    ground.py     the one ground detector and height map (fill)

A piece is the nodes sharing a DA3 frame -- a connected component of the
scene's edges, derived rather than stored. Each arrives internally
consistent but individually placed; everything here decides where it
actually sits, and writes the answer back onto its nodes.
"""
