"""Place, clean and fill a reconstructed scene; lay the world around it.

The fill or the world failing leaves the scene as it was before them: the placed scene is
still a result, and a map download failing costs only the land around.
"""
from streetview_to_3d.models.Segmenter import Segmenter
from streetview_to_3d.postprocess.Settings import Settings
from streetview_to_3d.postprocess.Stopwatch import Stopwatch
from streetview_to_3d.postprocess.clean.drop_blobs import drop_blobs
from streetview_to_3d.postprocess.clean.drop_sheets import drop_sheets
from streetview_to_3d.postprocess.fill.main import fill
from streetview_to_3d.postprocess.photos import photos
from streetview_to_3d.postprocess.place import place
from streetview_to_3d.postprocess.world.main import world as lay_world


def postprocess(scene, settings: Settings, labeller: Segmenter):
    """(scene, ground, world): the scene placed and finished, the fill's SceneGround and
    the World around it (None where not made). labeller: what labels the panos' photos
    the walk left unlabelled, off the GPU."""
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
            scene = photos(scene, labeller)
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
