"""Correcting what GPS leaves behind, using the road itself.

GPS places a piece to within a couple of metres and says nothing at all
about the heading of a piece built from one or two panoramas. The seams
that remain are visible: a step in height, a slope that does not carry
across, a piece rotated off the road.

The correction comes from the walking graph. Google's dots record where
the camera car actually drove, so chaining them into roads gives a line
per street that is smoother than any single piece's GPS and complete over
the whole piece. Each piece is seated on the line of the road it drove.

    road_mask                  rasterise a piece top-down and pick out the
                               grey patch its cameras stand on
    road_surface               that mask reduced to road points and a thin
                               kerb line
    road_frames                the roads, one frame each, and which piece
                               lies on which
    node_center_to_road_line   place every piece's camera nodes on the line
                               of the road it drove
    align_slope_of_pieces      then set every height and tilt against one
                               shared surface
    run                        the stages in order

The road line is used to MEASURE, never as something to match against: it
is GPS-derived, so fitting a piece's own GPS to it could only ever return
the piece to where GPS already put it. What it supplies is a direction of
travel and a smoother estimate of the line the car drove.
"""
