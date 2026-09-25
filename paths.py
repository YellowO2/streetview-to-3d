"""Where things live, resolved once from the project root rather than by
counting directories up from each module's own __file__, which breaks the
moment a file moves depth.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")     # fetched panoramas
SPLATS_DIR = os.path.join(PROJECT_ROOT, "splats")     # reconstruction runs

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SPLATS_DIR, exist_ok=True)
