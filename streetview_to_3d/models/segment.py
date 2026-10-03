"""Mark cars, people, thin poles/signs and water in a pano, so DA3 never turns them into points.

Movers ghost when panos merge, DA3 smears thin things into streaks and lays water at street
height. Each pano is segmented once, whole (Cityscapes SegFormer, plus an ADE20K one for water);
each DA3 view cuts its part of that mask, and the class map is kept with the scene for the fill.
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
# Cityscapes' 19 classes by id (get_segmenter checks), so a saved class map reads without the model
CITYSCAPES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
              "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
              "motorcycle", "bicycle")
CLASSES = CITYSCAPES + ("water",)
LABEL_IDS = {name: i for i, name in enumerate(CLASSES)}
WATER_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
WATER_CLASSES = ("water", "sea", "river", "lake", "swimming pool")   # ADE20K's
# What is dropped by default; any of CLASSES can be named per run instead.
DROP = MOVERS + THIN + ("water",)
# Grow each mask by a few pixels: depth at an object's edge smears between
# it and what's behind, and those in-between points are the worst floaters.
GROW_PX = 3
BATCH = 8   # images per pass; 16 ran a bigger model out of GPU memory
# The masker's input width, the image's own shape kept (None: the
# processor's square 512 x 512, which stretched the image)
MASK_W = 1024
# A pole is dropped only as a long straight stick: POLE_LONG times taller than wide, and
# POLE_STRAIGHT of its rows within POLE_TOL widths of one line (bollards, pillars are kept).
POLE_LONG, POLE_STRAIGHT, POLE_TOL = 4.0, 0.8, 1.0

_models = {}


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load(model_id, device):
    """(processor, model, {class name: id}), each built once per device."""
    if (model_id, device) not in _models:
        from transformers import AutoModelForSemanticSegmentation, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(device).eval()
        _models[(model_id, device)] = (processor, model,
                                       {name: int(i) for i, name in model.config.id2label.items()})
    return _models[(model_id, device)]


def get_segmenter(model_id=None, device=None):
    """(processor, model, {class name: id}) for a Cityscapes SegFormer
    (default MODEL_ID) on device (default: the GPU if there is one), the
    water one (WATER_MODEL_ID) loaded beside it: the default is built at
    startup on a Space (see streetview_to_3d.models.gpu), anything else on first
    use. Outside a GPU call on a Space, ask for "cpu"."""
    model_id, device = model_id or MODEL_ID, device or _device()
    processor, model, label_ids = _load(model_id, device)
    if label_ids != {n: i for i, n in enumerate(CITYSCAPES)}:
        raise ValueError(f"{model_id} does not label as Cityscapes does: {label_ids}")
    _load(WATER_MODEL_ID, device)
    return processor, model, label_ids


def _segment(images, processor, model):
    """One class-id map per image, each fed at its own shape, MASK_W wide."""
    import torch
    w, h = images[0].size
    size = {"width": MASK_W, "height": max(32, round(MASK_W * h / w / 32) * 32)} if MASK_W else None
    with torch.inference_mode():
        inputs = processor(images=images, return_tensors="pt", **({"size": size} if size else {})).to(model.device)
        labels = processor.post_process_semantic_segmentation(
            model(**inputs), target_sizes=[im.size[::-1] for im in images])
    return [lab.cpu().numpy() for lab in labels]


def label_views(paths, model_id=None, device=None):
    """(one class-id map per image path -- Cityscapes', water marked over
    it -- and {class name: id}, as LABEL_IDS). model_id, device: see
    get_segmenter."""
    device = device or _device()
    processor, model, _ = get_segmenter(model_id, device)
    w_processor, w_model, w_ids = _load(WATER_MODEL_ID, device)
    water = [w_ids[n] for n in WATER_CLASSES]
    out = []
    for start in range(0, len(paths), BATCH):
        images = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH]]
        for lab, wet in zip(_segment(images, processor, model), _segment(images, w_processor, w_model)):
            lab = lab.astype(np.uint8)
            lab[np.isin(wet, water)] = LABEL_IDS["water"]
            out.append(lab)
    return out, LABEL_IDS


def masks_from_labels(labels, label_ids, classes=None, long_only=True):
    """One boolean mask per class map, True on the classes to drop (DROP by
    default; poles only where long_poles says so, unless not long_only),
    grown by GROW_PX."""
    unknown = set(classes or ()) - set(label_ids)
    if unknown:
        raise ValueError(f"not classes we label: {sorted(unknown)}")
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


def pano_labels(path, model_id=None, device=None, saved=None):
    """A whole pano's class map (ids as in CLASSES), kept in saved (default
    beside it, named by the model) for whatever needs it later -- the scene
    carries a copy (labels_path). A saved one is read, not redone: the walk
    runs DA3 on one pano many times. model_id, device: see get_segmenter."""
    saved = saved or f"{path}.{(model_id or MODEL_ID).split('/')[-1]}+water.labels.png"
    if os.path.exists(saved):
        return np.asarray(Image.open(saved))
    labels = label_views([path], model_id, device)[0][0]
    os.makedirs(os.path.dirname(saved) or ".", exist_ok=True)
    Image.fromarray(labels).save(saved)
    return labels


def labels_path(scene_dir, pano_id):
    """Where a scene keeps a pano's class map."""
    return os.path.join(scene_dir, "labels", f"{pano_id}.png")


def pano_mask(labels, classes=None):
    """masks_from_labels for one pano's class map."""
    return masks_from_labels([labels], LABEL_IDS, classes)[0]


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


def drop_in_views(paths, panos, hfov, model_id=None, classes=None, device=None):
    """One mask per DA3 view (panoramic_da3's "pano_{i}_da3_{yaw}_0" files,
    pano i being panos[i]), cut from its pano's mask: each pano is
    segmented once, whole (pano_labels), and its class map kept."""
    masks, whole = [], {}
    for p in paths:
        i, yaw = map(int, _VIEW_NAME.match(os.path.basename(p)).groups())
        if i not in whole:
            whole[i] = pano_mask(pano_labels(panos[i], model_id, device), classes)
        masks.append(in_view(whole[i], yaw, hfov, *Image.open(p).size))
    return masks
