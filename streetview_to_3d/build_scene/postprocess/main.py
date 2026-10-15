"""Place, clean and fill a reconstructed scene; lay the world around it.

The fill or the world failing leaves the scene as it was before them: the placed scene is
still a result, and a map download failing costs only the land around.
"""
from streetview_to_3d.build_scene.postprocess.Settings import Settings
from streetview_to_3d.build_scene.postprocess.Stopwatch import Stopwatch
from streetview_to_3d.build_scene.postprocess.clean.drop_blobs import drop_blobs
from streetview_to_3d.build_scene.postprocess.clean.drop_sheets import drop_sheets
from streetview_to_3d.build_scene.postprocess.fill.main import fill
from streetview_to_3d.build_scene.postprocess.photos import photos
from streetview_to_3d.build_scene.postprocess.place import place
from streetview_to_3d.build_scene.postprocess.world.main import world as lay_world


def postprocess(scene, settings: Settings):
    """(scene, ground, world): the scene placed and finished, the fill's SceneGround and
    the World around it (None where not made); each node's photo carries the class map the
    walk made."""
    watch = Stopwatch()
    scene = place(scene)
    watch.lap("placement")
    ground = world = None
    if settings.fill:
        scene = drop_blobs(scene)
        watch.lap("floating bits")
        if settings.clean:
            scene = drop_sheets(scene)
            watch.lap("loose sheets")
        try:
            scene = photos(scene)
            scene, ground = fill(scene)
        except Exception as failure:
            watch.skipped("fill", failure)
        watch.lap("fill")
        try:                             # after the fill: its panos colour the buildings
            scene, world = lay_world(scene, ground, settings.world)
        except Exception as failure:
            watch.skipped("terrain", failure)
        watch.lap("terrain")
    return scene, ground, world


if __name__ == "__main__":
    # Finish a saved run's folder again, in place:
    #   python -m streetview_to_3d.build_scene.postprocess.main --dir data/runs/<run id>
    #       [--merge out.ply] [--no-fill] [--no-clean] [--google [--as-is]]
    import argparse
    import logging
    import os

    import numpy as np

    from streetview_to_3d.load import load
    from streetview_to_3d.build_scene.postprocess.world.Settings import Settings as WorldSettings
    from streetview_to_3d.build_scene.save import save
    from streetview_to_3d.tools.ply import write_ply

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description="Finish a reconstruction run's folder in place: place, clean, fill and lay the world.")
    ap.add_argument("--dir", required=True, help="a reconstruction run's directory")
    ap.add_argument("--merge", help="also write one combined .ply here")
    ap.add_argument("--no-fill", action="store_true", help="place only, no gap fill")
    ap.add_argument("--no-clean", action="store_true", help="leave the loose sheets (drop_sheets.py) in")
    ap.add_argument("--google", action="store_true",
                    help="merge the scene into the google.ply beside it (the viewer's ?google-save)")
    ap.add_argument("--as-is", action="store_true", help="with --google: leave DA3's points unbent")
    args = ap.parse_args()

    run_dir = os.path.expanduser(args.dir)
    world_settings = WorldSettings(google=os.path.join(run_dir, "google.ply") if args.google else None,
                                   bend=not args.as_is)
    finished, ground, laid = postprocess(load(run_dir), Settings(fill=not args.no_fill, clean=not args.no_clean,
                                                                 world=world_settings))
    save(finished, ground, laid, run_dir)
    if args.merge:
        # every placed node's points, in the world, in one .ply, for inspection
        out = os.path.expanduser(args.merge)
        ready = finished.placed()
        if not ready:
            raise ValueError("no node has been placed yet -- run the postprocess first")
        skipped = sum(1 for n in finished.nodes if n.ply and not n.transform)
        if skipped:
            logging.getLogger(__name__).warning(f"{skipped} node(s) have points but no transform, skipped")
        pts = np.concatenate([n.cloud.world(n.transform) for n in ready])
        cols = np.concatenate([n.cloud.colours for n in ready])
        write_ply(out, pts, cols)
        logging.getLogger(__name__).info(f"wrote {len(pts):,} points from {len(ready)} node(s) to {out}")
