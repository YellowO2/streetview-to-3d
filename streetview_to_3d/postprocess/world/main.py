"""The world round a placed scene: land, roads, buildings, water and life.

From free global maps (OSM, AWS Terrain Tiles, Google's and Sentinel-2's imagery, the JRC water
map) and Google's pano elevations, downloaded first and kept, coloured from the panos near the
cameras; with settings.google, merged into Google's 3D Tiles. Saved (save.py) as land.ply,
terrain.ply, roads.ply, buildings.ply, blocks.ply, water.json and life.json (and google.ply).
"""
from streetview_to_3d.postprocess.world.Elevation import Elevation
from streetview_to_3d.postprocess.world.HeightMap import HeightMap
from streetview_to_3d.postprocess.world.OpenStreetMap import OpenStreetMap
from streetview_to_3d.postprocess.world.Satellite import Satellite
from streetview_to_3d.postprocess.world.Settings import Settings
from streetview_to_3d.postprocess.world.Site import Site
from streetview_to_3d.postprocess.world.Water import Water
from streetview_to_3d.postprocess.world.WaterMap import WaterMap
from streetview_to_3d.postprocess.world.buildings.main import buildings
from streetview_to_3d.postprocess.world.car_roads import car_roads
from streetview_to_3d.postprocess.world.driving_side import driving_side
from streetview_to_3d.postprocess.world.elevations_around import elevations_around
from streetview_to_3d.postprocess.world.google.main import google
from streetview_to_3d.postprocess.world.google_points import google_points
from streetview_to_3d.postprocess.world.land.main import land
from streetview_to_3d.postprocess.world.land_colours import land_colours
from streetview_to_3d.postprocess.world.life import life
from streetview_to_3d.postprocess.world.outlines.main import outlines
from streetview_to_3d.postprocess.world.pano_views import pano_views
from streetview_to_3d.postprocess.world.roads.main import roads
from streetview_to_3d.postprocess.world.scene_files import scene_files
from streetview_to_3d.postprocess.world.water_outline import water_outline
from streetview_to_3d.postprocess.world.world_files import world_files


def world(scene, ground, settings: Settings = Settings()):
    """(scene, world): the scene naming its world's files, and the World laid round it (None
    without a placed camera). ground: the fill's SceneGround (None: none)."""
    site = Site(scene, ground)
    if not len(site.cameras):
        return scene, None
    osm = OpenStreetMap(site)
    elements = osm.elements()
    heights = HeightMap()
    elevations = elevations_around(site)
    water_map = WaterMap(site, heights)
    satellite = Satellite(site)
    views = pano_views(scene)
    side = driving_side(site)
    elevation = Elevation(site, heights, elevations, elements)
    shoreline = water_outline(elements, site, osm, water_map)
    water = Water(site, elevation, shoreline, water_map)
    streets = roads(site, elements, elevation, water, views)
    footprints = outlines(elements, site, elevation)
    landform = land(site, elevation, streets, water, footprints)
    landform = land_colours(landform, site, water, satellite, views)
    standing = buildings(footprints, landform, site, satellite, views)
    stretches = car_roads(elements, streets, footprints, site)
    moving = life(site, stretches, water, elevation, side)
    world = world_files(landform, streets, standing, water, moving)
    scene = scene_files(scene, world)
    if settings.google:
        google_tiles = google_points(settings.google)
        scene, world = google(scene, site, world, google_tiles, settings)
    return scene, world


if __name__ == "__main__":
    # Lay the world round a saved scene again, and save it there; --google merges the scene
    # into the google.ply saved beside it (the viewer's ?google-save), --as-is leaves DA3's
    # points unbent:
    #   python -m streetview_to_3d.postprocess.world.main SCENE_DIR [--google [--as-is]]
    import logging
    import os
    import sys

    from streetview_to_3d.load import load
    from streetview_to_3d.postprocess.SceneGround import SceneGround
    from streetview_to_3d.postprocess.photos import photos
    from streetview_to_3d.save import save

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    folders = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(folders) != 1:
        sys.exit("    python -m streetview_to_3d.postprocess.world.main SCENE_DIR\n"
                 "    python -m streetview_to_3d.postprocess.world.main SCENE_DIR --google [--as-is]")
    folder = folders[0]
    world_settings = Settings(google=os.path.join(folder, "google.ply") if "--google" in sys.argv else None,
                              bend="--as-is" not in sys.argv)
    scene, ground = photos(load(folder)), SceneGround.load(folder)
    scene, laid = world(scene, ground, world_settings)
    save(scene, ground, laid, folder)
