"""The world round the scene from free maps: land, roads, buildings, water and life.

main.py wires it; its loaders (OpenStreetMap, HeightMap, elevations_around, WaterMap, Satellite,
pano_views, driving_side, google_points) download or read what it needs first, each keeping its
own cache; postprocess/world/TileMap.py reads the tile maps.

    python -m streetview_to_3d.build.postprocess.world.main SCENE_DIR [--google [--as-is]]
"""
