# TODO

- Mask cars/people: done, on the public Space (app d6bc37b). Leaves holes on the road.
- Per-date pano poses: done (1d9ac49). Older dates used to take the newest capture's GPS and heading -- up to 7 m and 47 deg off.
- Google base (google_base/, see ARCHITECTURE step 5): done. Exact planes per pano, dense part only, walls merged across panos, one ground rebuilt as an even grid. ~4 s for 8 panos. Not yet used by the pipeline.
- Next: put DA3 on the Google base, fill DA3's gaps from it, paint once. Tried in scratch scripts, not in the repo yet:
  1. Fit each DA3 piece or node onto the base: turn + shift only (scale locked to the GPS fit), at most 1.5 m / 15 deg. Snapping each node to its own GPS first, ignoring DA3's links, was about as good as the linked fit on Stockholm (Google nodes 11-24 cm from the base).
  2. Gentle ground unroll: a smooth up/down map (5 m grid) from DA3 ground vs Google ground, everything above rides along. Too fine a grid (2 m) squashed bushes.
  3. Merge: keep Google points only where DA3, seen from each pano, has nothing at or in front of them.
  4. Paint everything once from the fewest scene panos (greedy: the pano seeing most unpainted points cleanly goes first), skipping masked pixels and each pano's own car.
  Scaling DA3 from its ground vs Google's was unreliable (Stockholm nodes 0.92-1.94). Apple nodes can stay 1-1.6 m off Google even after the fit.
- Floor-hole fill (reconstruct.ground_fill, on the Space): superseded once the base is in the pipeline.
- DA3 on forward-facing views only (180° instead of 360°): may link better, and halves the views.
- Apple depth is poor: neighbours keep 0/12 views, target 3/12.
- Splats from several panos (future work).
- Mapillary as an extra image source where Google/Apple have none.
