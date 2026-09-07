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
DATA_DIR = os.path.join(PROJECT_ROOT, "data")         # committed inputs
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")       # generated pages

# Where exported piece sets live: one directory per selection, holding
# piece_*.ply, piece_*_meta.json and, once solved, piece_transforms.json.
# Overridable so the same code can read a local set or a mounted one.
PIECES_DIR = os.environ.get("PIECES_DIR", os.path.join(PROJECT_ROOT, "pieces"))

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SPLATS_DIR, exist_ok=True)
