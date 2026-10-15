"""The names of the files the world is saved as, beside scene.json."""

LAND = "land.ply"
TERRAIN = "terrain.ply"
ROADS = "roads.ply"
BUILDINGS = "buildings.ply"
BLOCKS = "blocks.ply"
WATER = "water.json"
LIFE = "life.json"
GOOGLE = "google.ply"
# Scene's field for each, in the order the scene names them
FIELDS = (("land", LAND), ("roads", ROADS), ("terrain", TERRAIN), ("water", WATER), ("life", LIFE),
          ("buildings", BUILDINGS), ("blocks", BLOCKS))
