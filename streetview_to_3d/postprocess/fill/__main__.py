"""usage: python -m streetview_to_3d.postprocess.fill SCENE_DIR [OUT_DIR]

Fills a placed scene in place, or a copy of it in OUT_DIR."""
import os
import shutil
import sys

from streetview_to_3d.postprocess.fill import run


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) == 3 else src
    if out != src:
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(src):
            if f.endswith(".ply") or f == "scene.json":
                shutil.copy(os.path.join(src, f), os.path.join(out, f))
    run(out)


if __name__ == "__main__":
    main()
