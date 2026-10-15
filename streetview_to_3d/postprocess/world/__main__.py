"""Lay the world round a saved scene again, and save it there.

    python -m streetview_to_3d.postprocess.world SCENE_DIR
    python -m streetview_to_3d.postprocess.world SCENE_DIR --google [--as-is]

--google merges the scene into the google.ply saved beside it (the viewer's ?google-save);
--as-is leaves DA3's points unbent.
"""
import logging
import os
import sys


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    folders = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(folders) != 1:
        sys.exit(__doc__.split("\n\n")[1])
    from streetview_to_3d.load import load
    from streetview_to_3d.postprocess.SceneGround import SceneGround
    from streetview_to_3d.postprocess.photos import photos
    from streetview_to_3d.postprocess.world.Settings import Settings
    from streetview_to_3d.postprocess.world.main import world
    from streetview_to_3d.save import save
    folder = folders[0]
    settings = Settings(google=os.path.join(folder, "google.ply") if "--google" in sys.argv else None,
                        bend="--as-is" not in sys.argv)
    scene, ground = photos(load(folder)), SceneGround.load(folder)
    scene, laid = world(scene, ground, settings)
    save(scene, ground, laid, folder)


if __name__ == "__main__":
    main()
