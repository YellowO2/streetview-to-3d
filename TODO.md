# TODO

- Mask cars/people: done, on the public Space (app d6bc37b). Leaves holes on the road.
- Per-date pano poses: done (1d9ac49). Older dates used to take the newest capture's GPS and heading -- up to 7 m and 47 deg off.
- Google base (google_base/, ARCHITECTURE step 5): done.
- DA3: Nested 1.1 everywhere, 90 deg views, Google only (Apple off: GPS 1-1.6 m off, poor depth). Links where it can; unlinked places keep their best pano placed by GPS + heading. Solo mode (the "Link panoramas" toggle off) also adds Google's neighbour panos.
- Masker: SegFormer-B2 (B5 ran out of GPU memory); drops people, vehicles (not trains), poles, traffic lights and signs, where every view that sees a spot agrees. Views fed unstretched, 1024 wide. Model and class list settable per run.
- Fill (fill/, ARCHITECTURE step 6): done -- one ground, Google's walls in DA3's gaps (DA3 wins where both), colour from the nearest pano.
- Not done: walls of neighbouring panos are not merged with each other (DA3 vs DA3); rebuilding thin poles as clean cylinders.
- DA3 on forward-facing views only (180° instead of 360°): may link better, and halves the views.
- Splats from several panos (future work).
- Mapillary as an extra image source where Google/Apple have none.
