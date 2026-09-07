"""Look at what road alignment actually saw for particular pieces.

    python -m tests.inspect_road_align --pieces 3,9,5 --out /tmp/inspect.png

For each piece: the top-down photo it works from, and the road it
extracted from that photo. Renders every piece over the SAME bounds, so
the panels are directly comparable and you can see where each sits.
"""
import argparse
import os

import numpy as np
from PIL import Image

from alignment.road import CELL, direction_from_shape, road_cells, road_direction
from alignment.run_road_align import load_pieces


def panel(img, mask):
    """Photo, and photo with the detected road tinted red."""
    photo = (img * 255).astype(np.uint8)
    over = photo.copy()
    over[mask] = (0.35 * over[mask] + 0.65 * np.array([255, 40, 40])).astype(np.uint8)
    prep = lambda a: np.transpose(a, (1, 0, 2))[::-1]   # north up
    return prep(photo), prep(over)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/gap3_joined")
    ap.add_argument("--pieces", required=True, help="comma-separated ids")
    ap.add_argument("--out", default="/tmp/inspect_road_align.png")
    ap.add_argument("--cell", type=float, default=CELL)
    ap.add_argument("--margin", type=float, default=25.0)
    args = ap.parse_args()

    ids = [int(x) for x in args.pieces.split(",")]
    fits, clouds = load_pieces(args.dir)
    cams = np.vstack([fits[i]["cams"] for i in ids])
    bounds = (cams[:, 0].min() - args.margin, cams[:, 0].max() + args.margin,
              cams[:, 1].min() - args.margin, cams[:, 1].max() + args.margin)
    print(f"shared bounds {bounds[1]-bounds[0]:.0f} x {bounds[3]-bounds[2]:.0f} m\n")

    from alignment.road import top_down
    rows, masks = [], {}
    for i in ids:
        img, occ = top_down(*clouds[i], bounds, args.cell)
        mask, road, white = road_cells(*clouds[i], bounds, args.cell)
        masks[i] = mask
        deg, agree, sharp = road_direction(road, white)
        f = fits[i]
        r = "n/a" if f["resid"] is None else f"{f['resid']:.2f}m"
        a = "no paint" if agree is None else f"{agree:+.1f} deg"
        print(f"piece_{i}: {f['n']} node(s), GPS residual {r}, "
              f"{int(mask.sum())} road cell(s)")
        print(f"   road direction {deg:.1f} deg, sharpness +-{sharp/2:.0f} deg, "
              f"white line agrees within {a}")
        rows.append((i, *panel(img, mask)))

    print("\nshared road cells between pieces:")
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            n = int((masks[i] & masks[j]).sum())
            print(f"   piece_{i} & piece_{j}: {n} cell(s) = {n*args.cell**2:.0f} m2")

    h = rows[0][1].shape[0]
    w = rows[0][1].shape[1]
    canvas = np.zeros((len(rows) * (h + 8), w * 2 + 8, 3), np.uint8)
    for k, (_, photo, over) in enumerate(rows):
        y = k * (h + 8)
        canvas[y:y + h, :w] = photo
        canvas[y:y + h, w + 8:w + 8 + w] = over
    Image.fromarray(canvas).save(args.out)
    print(f"\nwrote {args.out}  (rows: {', '.join(f'piece_{i}' for i in ids)}; "
          f"left = photo, right = road in red)")


if __name__ == "__main__":
    main()
