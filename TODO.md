# TODO

- Mask cars/people: SegFormer-B0 (Cityscapes) on DA3's views, drop masked pixels at backprojection. Tried on one pano: works, misses odd shapes (crane arm).
- Colour from fewer panos: give each point the colour of one pano (e.g. the nearest), not a mix of dates and lighting.
- DA3 on forward-facing views only (180° instead of 360°): may link better, and halves the views.
- Apple depth is poor: neighbours keep 0/12 views, target 3/12.
- Splats from several panos (future work).
- Mapillary as an extra image source where Google/Apple have none.
