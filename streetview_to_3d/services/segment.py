"""Find cars, people, poles and signs in DA3's views, so their pixels never
become points.

Moving things are what ghost when panoramas are merged: the same car shows
up once per photo, in a different place each time. A street-scene
segmenter (SegFormer-B2 trained on Cityscapes, 27M parameters) marks them
on the whole pano, once; each DA3 view takes its part of that mask and
panoramic_da3 leaves those pixels out (its drop_mask). DA3 itself still
sees the whole view, so poses are unchanged. The fill colours by the same
saved mask. DA3's views masked one by one (every view that saw a spot
having to agree) were tried: a little cleaner on NTU, where the whole pano
takes part of a long walkway roof for a bus, but twice the code, and the
fill needs the whole pano anyway.

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
# Sky has no real distance: DA3 puts it on a dome 100-140 m out. Off by
# default -- the confidence floor (services.da3_ops.CONF_FLOOR) already
# cuts it, with everything else far; name it per run (mask_classes) to drop
# it by label instead, e.g. with the floor lowered. At a harbour the masker
# missed patches of sky that views disagreed on.
SKY = ("sky",)
# What is dropped by default. Any of Cityscapes' 19 classes can be named
# per run instead: road, sidewalk, building, wall, fence, pole, traffic
# light, traffic sign, vegetation, terrain, sky, person, rider, car, truck,
# bus, train, motorcycle, bicycle.
DROP = MOVERS + THIN
# Grow each mask by a few pixels: depth at an object's edge smears between
# it and what's behind, and those in-between points are the worst floaters.
GROW_PX = 3
BATCH = 8   # images per pass; 16 ran a bigger model out of GPU memory
# The masker's input width, the image's own shape kept (None: the
# processor's square 512 x 512, which stretched the image)
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


def pano_mask(path, model_id=None, classes=None, device=None, reuse=False):
    """drop_movers on a whole pano, saved beside it (path + ".mask.png")
    for the fill to colour by the same mask; reuse: load that instead when
    it is there."""
    saved = path + ".mask.png"
    if reuse and os.path.exists(saved):
        return np.asarray(Image.open(saved)) > 0
    m = drop_movers([path], model_id, classes, device)[0]
    Image.fromarray(m.astype(np.uint8) * 255).save(saved)
    return m


def in_view(pano, yaw, hfov, w, h):
    """The part of a pano-shaped mask a w x h view at yaw sees (hfov wide,
    level, as panoramic_da3's extract_views_for_da3 cuts them)."""
    H, W = pano.shape
    f = w / 2 / np.tan(np.radians(hfov) / 2)
    u, v = np.meshgrid((np.arange(w) - (w - 1) / 2) / f, (np.arange(h) - (h - 1) / 2) / f)
    t = np.radians(yaw)
    x, z = np.cos(t) * u + np.sin(t), -np.sin(t) * u + np.cos(t)
    lon, lat = np.arctan2(x, z), np.arctan2(v, np.hypot(x, z))
    px = np.round((lon / (2 * np.pi) + .5) * (W - 1)).astype(int) % W
    py = np.clip(np.round((lat / np.pi + .5) * (H - 1)).astype(int), 0, H - 1)
    return pano[py, px]


_VIEW_NAME = re.compile(r"^pano_(\d+)_da3_(-?\d+)_0\.\w+$")


def drop_in_views(paths, panos, hfov, **kw):
    """One mask per DA3 view (panoramic_da3's "pano_{i}_da3_{yaw}_0" files,
    pano i being panos[i]), cut from its pano's pano_mask: each pano is
    segmented once, whole. kw: see drop_movers."""
    masks, whole = [], {}
    for p in paths:
        i, yaw = map(int, _VIEW_NAME.match(os.path.basename(p)).groups())
        if i not in whole:
            whole[i] = pano_mask(panos[i], **kw)
        masks.append(in_view(whole[i], yaw, hfov, *Image.open(p).size))
    return masks
