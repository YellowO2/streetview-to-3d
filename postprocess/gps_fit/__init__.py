"""Putting DA3's output into the real world.

Two different questions, sharing one solver:

    discover_pieces  which nodes belong together as one piece -- a chunk
                     whose own nodes do not fit GPS is cut, and pieces are
                     grown outward only while everything still fits
    load_pieces      put an already-defined piece into world metres

`fit` holds the similarity fit both use, and the fixed origin that makes
every piece land in the same frame.
"""
