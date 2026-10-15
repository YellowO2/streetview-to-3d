"""usage: python -m streetview_to_3d.postprocess.fill SCENE_DIR [OUT_DIR]

Fills a placed scene in place, or a copy of it in OUT_DIR."""
import logging
import sys

from streetview_to_3d.load import load
from streetview_to_3d.postprocess.fill.main import fill
from streetview_to_3d.postprocess.photos import photos
from streetview_to_3d.save import save


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) == 3 else src
    scene, ground = fill(photos(load(src)))
    save(scene, ground, None, out)


if __name__ == "__main__":
    main()
