"""Correcting what GPS leaves behind, using the road itself.

GPS places pieces to within a couple of metres and says nothing at all
about the heading of a piece built from one panorama. The seams that
remain are visible: a step in height, a slope that does not carry across,
a piece rotated off the road.

    road, kerb                 find the road, and its edges
    extract_road_lines         reduce a piece to left kerb, centre, right
    camera_route               one direction of travel for the whole run
    fit_pieces_to_road         place every piece against one road
    seat_pieces_on_surface     then set every height and tilt against one surface
    run                        the stages in order
"""
