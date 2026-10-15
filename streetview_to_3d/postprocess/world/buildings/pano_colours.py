"""Buildings the panos see enough of take the colour they see, softened."""
import numpy as np

from streetview_to_3d.postprocess.world.buildings.tools.colours import soften

SEE_M = 150.0                 # panos colour buildings within this distance
BEHIND_M, BEHIND = 3.0, 0.05  # depth tolerance (m, plus share of distance) for a wall to count as seen
SEEN_MIN = 20                 # pixels a building needs to take the panos' colour
SAMPLE = 16                   # colour judged from every SAMPLE-th point
SAMPLE_HIDING = 4             # occluders: every SAMPLE_HIDING-th DA3 point


def pano_colours(blocks, outlines, colours, site, views):
    """(blocks, colours): each building DA3 reaches that the panos (views: PanoViews) see enough of
    takes its median colour in them, softened, in colours and on its points (Blocks) but where
    tagged roofs keep theirs; seen: how many. DA3's points (site: Site) hide what is behind them."""
    if not views.cameras:
        return blocks, colours
    colours = colours.copy()
    own = _seen(blocks.pts, blocks.which, len(outlines.items), views.cameras, views.photos, site.points)
    seen = ~np.isnan(own[:, 0]) & blocks.reached
    colours[seen] = np.array([soften(c) for c in own[seen]]).reshape(-1, 3)
    recolour = seen[blocks.which] & ~blocks.own               # tagged roof colours stay
    blocks.cols[recolour] = colours[blocks.which[recolour]]
    blocks.seen = int(seen.sum())
    return blocks, colours


def _seen(pts, which, n, cameras, photos, occluders):
    """(n, 3) each building's median colour in the panos, NaN where under SEEN_MIN pixels are seen.

    A point is seen by a camera within SEE_M if nothing is more than BEHIND_M
    (+ BEHIND x distance) in front of it (so DA3's own copy of the wall does
    not hide it) and its pixel is labelled building or wall."""
    from streetview_to_3d.models.Segmenter import LABEL_IDS
    from streetview_to_3d.postprocess.tools.paint import NADIR_DEG, depth_buffer, pano_at, pixel
    ids = [LABEL_IDS["building"], LABEL_IDS["wall"]]
    pts, which = pts[::SAMPLE], which[::SAMPLE]
    everything = np.concatenate([occluders[::SAMPLE_HIDING], pts])
    samples, owners = [], []
    for cam, ph in zip(cameras, photos):
        if ph is None:
            continue
        near = depth_buffer(cam, everything)
        u, v, r, below = cam.look(pts)
        seen = (r < SEE_M) & (below < NADIR_DEG) & (r <= near[pixel(u, v)] + BEHIND_M + BEHIND * r)
        seen &= np.isin(pano_at(ph[1], u, v), ids) & ~pano_at(ph[2], u, v)
        if seen.any():
            from PIL import Image
            img = np.asarray(Image.open(ph[0]).convert("RGB"))
            samples.append(pano_at(img, u[seen], v[seen]) / 255.0)
            owners.append(which[seen])
    out = np.full((n, 3), np.nan)
    if not samples:
        return out
    samples, owners = np.concatenate(samples), np.concatenate(owners)
    order = np.argsort(owners, kind="stable")
    samples, owners = samples[order], owners[order]
    starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(owners)]):
        if b - a >= max(3, SEEN_MIN / SAMPLE):
            out[owners[a]] = np.median(samples[a:b], 0)
    return out
