"""Placing what was reconstructed: separate clouds into one scene. No GPU.

    pipeline.py   the whole thing as one call
    gps_fit/      fit each piece to its own GPS, and split it where that fails
    road_align/   seat every piece on its road and on the real ground

A piece arrives internally consistent but individually placed; everything
here decides where it actually sits.
"""
