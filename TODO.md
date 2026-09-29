# TODO

- Mask cars/people: done, on the public Space (app d6bc37b). Leaves holes on the road.
- Per-date pano poses: done (1d9ac49). Older dates used to take the newest capture's GPS and heading -- up to 7 m and 47 deg off.
- DA3: Nested 1.1 everywhere, 90 deg views, Google only (Apple off: GPS 1-1.6 m off, poor depth). Links where it can; unlinked places keep their best pano placed by GPS + heading. Solo mode (the "Link panoramas" toggle off) also adds Google's neighbour panos.
- Masker: SegFormer-B2 (B5 ran out of GPU memory); drops people, vehicles (not trains), long straight poles, traffic lights and signs. Each pano masked once, whole (1024 wide), saved; DA3's views cut from it, the fill colours by it. Model and class list settable per run.
- Fill (fill/, ARCHITECTURE step 5): done -- one ground, colour from the nearest pano. Google's depth maps removed: the surroundings (step 6) replace them.
- Surroundings (postprocess/terrain.py, ARCHITECTURE step 6): land, roads and OSM buildings, fitted to DA3 and coloured from the panos.
- Not done: walls of neighbouring panos are not merged with each other (DA3 vs DA3); rebuilding thin poles as clean cylinders.
- DA3 on forward-facing views only (180° instead of 360°): may link better, and halves the views.
- Splats from several panos (future work).
- Mapillary as an extra image source where Google/Apple have none.
