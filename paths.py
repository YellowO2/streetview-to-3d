import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(PROJECT_ROOT, "images")
SPLATS_DIR = os.path.join(PROJECT_ROOT, "splats")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(SPLATS_DIR, exist_ok=True)
