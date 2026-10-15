"""How far the world reaches, and how it asks the maps."""

RADIUS_M = 1000.0                          # the land's reach (the viewer's haze is total by then)
OSM_M, ROADS_M = RADIUS_M, 700.0           # OSM reach for buildings/water and for main roads
NEAR_M = 50.0                              # roads and bridges are points this near a camera, meshes past
M_PER_LAT = 111320.0                       # metres per degree of latitude
FULL_M = 250.0            # map data: everything this near the cameras
FAR_BUILDING_M = 80.0     # ... past FULL_M, only buildings with at least this perimeter
USER_AGENT = "streetview-to-3d (https://github.com/YellowO2/streetview-to-3d)"   # how the maps are asked
