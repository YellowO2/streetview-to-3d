"""Placing what was reconstructed: separate clouds into one scene. No GPU.

    pipeline.py   the whole thing as one call
    gps_fit/      fit each piece to its own GPS
    road_align/   seat every piece on its road and on the real ground

A piece is the nodes sharing a DA3 frame -- a connected component of the
scene's edges, derived rather than stored. Each arrives internally
consistent but individually placed; everything here decides where it
actually sits, and writes the answer back onto its nodes.
"""
