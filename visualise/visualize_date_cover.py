"""Quick visual sanity check for the whole-NTU date cover -- one dot per
corridor point, colored by which real capture date build_date_cover
assigned it (see street_builder/build_graph/global_dates.py). Dots with
no real candidate on any date (never assigned) are shown in light gray.

Usage:
    python -m tests.visualize_date_cover
"""
import json
import os

import matplotlib.pyplot as plt

NTU_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ntu")


def main():
    with open(os.path.join(NTU_DIR, "fetch_metadata.json")) as f:
        points = json.load(f)["points"]
    with open(os.path.join(NTU_DIR, "date_cover.json")) as f:
        cover = {int(k): v for k, v in json.load(f).items()}

    dates = sorted({d for d in cover.values()})
    cmap = plt.get_cmap("tab20", max(len(dates), 1))
    color_by_date = {date: cmap(i) for i, date in enumerate(dates)}

    fig, ax = plt.subplots(figsize=(12, 12))

    unassigned = [points[i] for i in range(len(points)) if i not in cover]
    if unassigned:
        lats, lons = zip(*unassigned)
        ax.scatter(lons, lats, c="lightgray", s=8, label=f"unassigned ({len(unassigned)})")

    for date in dates:
        dots = [points[i] for i, d in cover.items() if d == date]
        lats, lons = zip(*dots)
        ax.scatter(lons, lats, c=[color_by_date[date]], s=10, label=f"{date} ({len(dots)})")

    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal")
    ax.set_title(f"NTU whole-corridor date cover -- {len(cover)}/{len(points)} dot(s) assigned, {len(dates)} date(s)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7, markerscale=2)
    fig.tight_layout()

    out_path = os.path.join(NTU_DIR, "date_cover_visualization.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
