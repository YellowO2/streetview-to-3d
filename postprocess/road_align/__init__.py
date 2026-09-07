"""Correcting what GPS leaves behind, using the road itself.

GPS places a piece to within a couple of metres and says nothing at all
about the heading of a piece built from one or two panoramas. The seams
that remain are visible: a step in height, a slope that does not carry
across, a piece rotated off the road.

The correction comes from the walking graph. Google's dots record where
the camera car actually drove, so chaining them into roads gives a line
per street that is smoother than any single piece's GPS and complete over
the whole piece. Each piece is seated on the line of the road it drove.

    road                       find the road surface in a piece
    road_frames                the roads, one frame each, and which piece
                               lies on which
    seat_on_road_line          place every piece on its road's line
    seat_pieces_on_surface     then set every height and tilt against one
                               shared surface
    run                        the stages in order

The road line is used to MEASURE, never as something to match against: it
is GPS-derived, so fitting a piece's own GPS to it could only ever return
the piece to where GPS already put it. What it supplies is a direction of
travel and a smoother estimate of the line the car drove.
"""
