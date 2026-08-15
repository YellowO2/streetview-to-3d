"""Experimental panorama slope de-tilting, used by app.py's DA3 point-cloud path."""


def correct_slope(image_path: str, heading: float, pitch: float, roll: float, multiplier: float = 1.0) -> str:
    """De-tilt a panorama by its own heading/pitch/roll (Street View's
    upright-correction metadata), so its ground plane looks level before DA3
    depth/pose inference. Experimental — validating whether this improves
    DA3's view-consistency filtering on sloped streets. `multiplier` scales
    pitch/roll, to test whether a stronger-than-reported correction helps
    further. Returns a new file path; original untouched."""
    import cv2
    from components.ViewExtractor.Equirec2Perspec import rotate_equirectangular

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    corrected = rotate_equirectangular(img, heading=heading, roll=roll * multiplier, pitch=pitch * multiplier)
    out_path = image_path.rsplit(".", 1)[0] + "_leveled.jpg"
    cv2.imwrite(out_path, corrected)
    return out_path
