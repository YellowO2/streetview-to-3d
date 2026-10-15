"""Remove floating bits or loose sheets from a saved scene, in place.

    python -m scripts.clean blobs SCENE_DIR
    python -m scripts.clean sheets SCENE_DIR
"""
import logging
import sys

from streetview_to_3d.load import load
from streetview_to_3d.build_scene.postprocess.clean.drop_blobs import drop_blobs
from streetview_to_3d.build_scene.postprocess.clean.drop_sheets import drop_sheets
from streetview_to_3d.build_scene.save import save

CLEANERS = {"blobs": drop_blobs, "sheets": drop_sheets}


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) != 3 or sys.argv[1] not in CLEANERS:
        sys.exit(__doc__.split("\n\n")[1])
    kind, folder = sys.argv[1:]
    save(CLEANERS[kind](load(folder)), None, None, folder)


if __name__ == "__main__":
    main()
