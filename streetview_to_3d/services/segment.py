"""Find cars, people, poles and signs in DA3's views, so their pixels never
become points.

Moving things are what ghost when panoramas are merged: the same car shows
up once per photo, in a different place each time. A street-scene
segmenter (SegFormer-B2 trained on Cityscapes, 27M parameters) marks them
per view; panoramic_da3 then leaves those pixels out (its drop_mask). DA3
itself still sees the whole view, so poses are unchanged.

Parked cars are dropped too -- the class can't tell them apart -- which
leaves a gap on the road that other panoramas usually fill.

Poles, traffic lights and signs go too: DA3 smears anything this thin into
a streak or a broken stick, and a missing street light reads better than a
wrong one. Cityscapes' "traffic light" and "traffic sign" are only the
light box and the board; every post, lamp posts included, is "pole".
"""
import os
import re

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# B2 caught poles and people B0 missed on Singapore; B5 ran out of GPU
# memory next to DA3. Any Cityscapes SegFormer (b0-b5) works per run.
MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
MOVERS = ("person", "rider", "car", "truck", "bus", "motorcycle", "bicycle")   # not train: it is part of the place
THIN = ("pole", "traffic light", "traffic sign")
# What is dropped by default. Any of Cityscapes' 19 classes can be named
# per run instead: road, sidewalk, building, wall, fence, pole, traffic
# light, traffic sign, vegetation, terrain, sky, person, rider, car, truck,
# bus, train, motorcycle, bicycle.
DROP = MOVERS + THIN
# Grow each mask by a few pixels: depth at an object's edge smears between
# it and what's behind, and those in-between points are the worst floaters.
GROW_PX = 3
BATCH = 8   # views per pass; 16 ran a bigger model out of GPU memory
# The masker's input width, the image's own shape kept (None: the
# processor's square 512 x 512, which stretched a 16:9 view)
MASK_W = 1024
# A pole is dropped only when it is a long straight stick (a lamp post, a
# sign's post): at least POLE_LONG times taller than it is wide, and
# POLE_STRAIGHT of its rows within POLE_TOL widths of one straight line --
# a lamp's arm is only a few rows, a bollard, a thick pillar or a curved
# pole is kept.
POLE_LONG, POLE_STRAIGHT, POLE_TOL = 4.0, 0.8, 1.0

_models = {}


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_segmenter(model_id=None, device=None):
    """(processor, model, {class name: id}) for a Cityscapes SegFormer
    (default MODEL_ID) on device (default: the GPU if there is one): the
    default is built at startup on a Space (see streetview_to_3d.gpu),
    anything else on first use. Outside a GPU call on a Space, ask for
    "cpu"."""
    model_id, device = model_id or MODEL_ID, device or _device()
    if (model_id, device) not in _models:
        from transformers import AutoModelForSemanticSegmentation, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
        label_ids = {name: int(i) for i, name in model.config.id2label.items()}
        _models[(model_id, device)] = (processor, model, label_ids)
    return _models[(model_id, device)]


def label_views(paths, model_id=None, device=None):
    """(one Cityscapes class-id map per image path, {class name: id}), each
    image fed at its own shape, MASK_W wide. model_id, device: see
    get_segmenter."""
    import torch
    processor, model, label_ids = get_segmenter(model_id, device)
    out = []
    for start in range(0, len(paths), BATCH):
        images = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH]]
        w, h = images[0].size
        size = {"width": MASK_W, "height": max(32, round(MASK_W * h / w / 32) * 32)} if MASK_W else None
        with torch.inference_mode():
            inputs = processor(images=images, return_tensors="pt", **({"size": size} if size else {})).to(model.device)
            labels = processor.post_process_semantic_segmentation(
                model(**inputs), target_sizes=[im.size[::-1] for im in images])
        out += [lab.cpu().numpy().astype(np.uint8) for lab in labels]
    return out, label_ids


def masks_from_labels(labels, label_ids, classes=None, long_only=True):
    """One boolean mask per class map, True on the classes to drop (DROP by
    default; poles only where long_poles says so, unless not long_only),
    grown by GROW_PX."""
    unknown = set(classes or ()) - set(label_ids)
    if unknown:
        raise ValueError(f"not Cityscapes classes: {sorted(unknown)}")
    names = classes or DROP
    others = [label_ids[n] for n in names if n != "pole"]
    masks = []
    for lab in labels:
        m = np.isin(lab, others)
        if "pole" in names:
            pole = lab == label_ids["pole"]
            m |= long_poles(pole) if long_only else pole
        masks.append(binary_dilation(m, iterations=GROW_PX) if m.any() else m)
    return masks


def long_poles(pole):
    """The parts of a pole mask that are long straight sticks (see POLE_LONG)."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(pole.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if h < 8:
            continue
        rows = lab[y:y + h, x:x + w] == k
        filled = rows.any(1)
        width = area / filled.sum()                       # its typical width per row
        if h < POLE_LONG * width:
            continue
        ys = np.flatnonzero(filled)
        cx = np.array([np.flatnonzero(r).mean() for r in rows[filled]])
        a, b = np.polyfit(ys, cx, 1)
        keep[k] = (np.abs(cx - (a * ys + b)) <= POLE_TOL * width).mean() >= POLE_STRAIGHT
    return keep[lab]


def drop_movers(paths, model_id=None, classes=None, device=None):
    """One boolean mask per image path, True on cars, people, poles and the
    like. model_id: another Cityscapes SegFormer (see get_segmenter);
    classes: the class names to drop instead of DROP; device: see
    get_segmenter."""
    return masks_from_labels(*label_views(paths, model_id, device), classes)


_VIEW_NAME = re.compile(r"^(.*)da3_(-?\d+)_0\.\w+$")


def drop_in_views(paths, hfov, **kw):
    """drop_movers for DA3's views (panoramic_da3's "{prefix}da3_{yaw}_0"
    files, hfov wide), a pixel dropped only where every view of the same
    pano that sees it drops it (agree). kw: see drop_movers."""
    return agree(drop_movers(paths, **kw), paths, hfov)


def agree(masks, paths, hfov):
    """masks, each pixel kept dropped only where every view of the same
    pano that sees it drops it. Each spot is seen by about 3 views; on node
    10 of NTU the real cars, people and signs were dropped by all of them,
    while the masker's mistakes (a long white walkway roof taken for a bus
    or truck, here and there) were each one or two views'."""
    views = {}
    for i, p in enumerate(paths):
        m = _VIEW_NAME.match(os.path.basename(p))
        if m:
            views.setdefault(os.path.join(os.path.dirname(p), m[1]), []).append((i, float(m[2])))
    out = [m.copy() for m in masks]
    for group in views.values():
        for i, yi in group:
            if not masks[i].any():
                continue
            h, w = masks[i].shape
            f = w / 2 / np.tan(np.radians(hfov) / 2)
            u, v = np.meshgrid((np.arange(w) - (w - 1) / 2) / f, (np.arange(h) - (h - 1) / 2) / f)
            for j, yj in group:
                d = np.radians((yi - yj + 180) % 360 - 180)
                if j == i or abs(d) >= np.radians(hfov):
                    continue
                # view i's pixel ray, turned by the yaw between them, into view j
                x, z = np.cos(d) * u + np.sin(d), -np.sin(d) * u + np.cos(d)
                front = z > 1e-6
                zs = np.where(front, z, 1)
                mj = cv2.resize(masks[j].astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                xj, yj_ = np.round(f * x / zs + (w - 1) / 2), np.round(f * v / zs + (h - 1) / 2)
                seen = front & (xj >= 0) & (xj <= w - 1) & (yj_ >= 0) & (yj_ <= h - 1)
                says = np.zeros((h, w), bool)
                says[seen] = mj[yj_[seen].astype(int), xj[seen].astype(int)] > 0
                out[i] &= says | ~seen   # a view that doesn't see the spot has no say
    return out
