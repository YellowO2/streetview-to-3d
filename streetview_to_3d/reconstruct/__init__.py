"""Panoramas in, point clouds out. The GPU stage.

    walk_graph.py     search the corridor, testing real DA3 edges between
                      panoramas, growing a piece from the ones that hold
    join_segments.py  bridge separately-grown pieces where DA3 can
    solo.py           the other mode: every Google pano on its own, no links
    build.py          the orchestrator the UI and the CLI both call

A piece ends where DA3 stopped agreeing, which is why more than one can
come out of a single run.
"""
