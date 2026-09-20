"""Where things live, resolved once from the project root.

Modules used to find these by counting directories up from their own
__file__, which quietly breaks the moment a file moves depth -- moving
gps.py one level deeper turned every path into <package>/ntu instead of
<project>/ntu, and eleven modules stopped importing at once.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")     # fetched panoramas
SPLATS_DIR = os.path.join(PROJECT_ROOT, "splats")     # reconstruction runs
NTU_DIR = os.path.join(PROJECT_ROOT, "ntu")           # fetched NTU metadata

# The three graphs, in the order they are produced. They are easy to confuse,
# so the names say which is which:
#
#   GOOGLE_GRAPH     what Street View has: every Google panorama in the area
#                    and how they link. The raw source.
#   FETCHED_GRAPH    that corridor resampled into evenly spaced dots (2852
#                    panoramas -> 1573 dots), plus a census of what imagery
#                    Google AND Apple have at each dot and on what dates.
#                    Metadata only -- key, source, id, lat, lon, date. No
#                    images. This is what date selection reads, and what
#                    postprocess uses for real walking adjacency.
#   SELECTED_GRAPH   the subset actually chosen to reconstruct.
GOOGLE_GRAPH = os.path.join(NTU_DIR, "google_graph.json")
FETCHED_GRAPH = os.path.join(NTU_DIR, "downsampled_and_fetched_graph.json")
SELECTED_GRAPH = os.path.join(NTU_DIR, "graph.json")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")         # committed inputs
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")       # generated pages

# Where exported piece sets live: one directory per selection, holding
# a scene.json, its clouds, and once solved piece_transforms.json.
# Overridable so the same code can read a local set or a mounted one.
PIECES_DIR = os.environ.get("PIECES_DIR", os.path.join(PROJECT_ROOT, "pieces"))

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SPLATS_DIR, exist_ok=True)
