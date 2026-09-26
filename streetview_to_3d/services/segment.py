"""Find cars and people in DA3's views, so their pixels never become points.

Moving things are what ghost when panoramas are merged: the same car shows
up once per photo, in a different place each time. A small street-scene
segmenter (SegFormer-B0 trained on Cityscapes, ~4M parameters) marks them
per view; panoramic_da3 then leaves those pixels out (its drop_mask). DA3
itself still sees the whole view, so poses are unchanged.

Parked cars are dropped too -- the class can't tell them apart -- which
leaves a gap on the road that other panoramas usually fill.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
MOVERS = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
# Grow each mask by a few pixels: depth at an object's edge smears between
# it and what's behind, and those in-between points are the worst floaters.
GROW_PX = 3
BATCH = 16

_model = None


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_segmenter():
    """(processor, model, class ids to drop): built at startup on a Space
    (see streetview_to_3d.gpu), on first use elsewhere."""
    global _model
    if _model is None:
        from transformers import AutoModelForSemanticSegmentation, AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_ID).to(_device()).eval()
        label_ids = {name: int(i) for i, name in model.config.id2label.items()}
        _model = (processor, model, [label_ids[n] for n in MOVERS])
    return _model


def drop_movers(paths):
    """One boolean mask per image path, True on cars, people and the like."""
    import torch
    processor, model, drop = get_segmenter()
    masks = []
    for start in range(0, len(paths), BATCH):
        images = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH]]
        with torch.inference_mode():
            inputs = processor(images=images, return_tensors="pt").to(model.device)
            labels = processor.post_process_semantic_segmentation(
                model(**inputs), target_sizes=[im.size[::-1] for im in images])
        for lab in labels:
            m = np.isin(lab.cpu().numpy(), drop)
            masks.append(binary_dilation(m, iterations=GROW_PX) if m.any() else m)
    return masks
