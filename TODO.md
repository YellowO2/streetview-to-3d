# TODO

- Mask cars/people: on the dev Space, works (cars, people gone, ~6% fewer points, ~1 s GPU); leaves holes on the road. Not yet on the public Space.
- Google depth maps (streetlevel download_depth, mirrored left-right vs the photo): metric, ground exact (camera 2.45 m). DA3 x ~1.2 = metres, and most links agree with GPS at ~1.2.
  1. Cut a link whose DA3 spacing disagrees with GPS (Stockholm 0-1 was 2.44). Tried: ~/Downloads/cut_links.patch; not better on Stockholm, since road alignment moves pieces off GPS anyway and one link left is too few for scale.
  2. Floor-hole fill: done (5e62ec5, dev Space). Geometry from Google's depth per pixel, colour from a neighbour. Left: end-of-street cameras keep a small hole (nothing else sees under them); the patch right under a camera can look blocky.
  3. Align DA3 to Google's ground (scale/tilt per pano). Experimental. The fill already measures scale per piece from Google's ground: 1.13-1.20, matching the GPS link scale.
  4. If GPS is trusted fully, snap panos to GPS + heading and skip road alignment, rather than doing both.
- Colour from fewer panos: give each point the colour of one pano (e.g. the nearest), not a mix of dates and lighting.
- DA3 on forward-facing views only (180° instead of 360°): may link better, and halves the views.
- Apple depth is poor: neighbours keep 0/12 views, target 3/12.
- Splats from several panos (future work).
- Mapillary as an extra image source where Google/Apple have none.
