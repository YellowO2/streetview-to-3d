
# Files structure should be the following:

# map_selection/candidates.py   → raw fetch primitives (Google/Apple)
# build_graph/                  → candidate gathering + graph + date ranking
# reconstruction/                → pure solve logic only (walk_graph.py, join_segments.py, generate.py) — zero build_graph/main.py imports
# main.py                       → orchestrator (build_graph + GPU call + save/join)
# map_selection/tab.py           → UI wiring, the only layer that touches everything
