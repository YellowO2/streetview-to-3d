"""The scene merged into Google's 3D Tiles, standing for the map's world.

Google's points (google.ply, saved by the viewer's ?google-save) are slid onto DA3's points
and given the panos' look; each pano's DA3 points are bent onto Google's shape (unless
settings.bend is off); Google's go only where they stand in front of what a pano saw and on
the water, which the map's water takes instead, at Google's height: behind DA3's points
Google's stay, backing them. The scene then names google.ply in place of the map-built land,
roads and buildings (their files stay).
"""
from streetview_to_3d.postprocess.world.Settings import Settings
from streetview_to_3d.postprocess.world.google.GooglePoints import GooglePoints
from streetview_to_3d.postprocess.world.google.aligned import aligned
from streetview_to_3d.postprocess.world.google.bent import bent
from streetview_to_3d.postprocess.world.google.google_scene import google_scene
from streetview_to_3d.postprocess.world.google.merged_world import merged_world
from streetview_to_3d.postprocess.world.google.pano_centres import pano_centres


def google(scene, site, world, google_tiles: GooglePoints, settings: Settings):
    """(scene, world) merged into Google's points; site: Site, world: the World laid round it."""
    centres = pano_centres(scene)
    google_tiles = aligned(google_tiles, site, centres)
    scene, clouds = bent(scene, centres, google_tiles, settings)
    world = merged_world(google_tiles, scene, clouds, centres, world)
    scene = google_scene(scene)
    return scene, world
