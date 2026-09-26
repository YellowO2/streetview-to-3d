"""Find cars, people and thin poles in DA3's views, so their pixels never
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
wrong one.
"""
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation

# B2 caught poles and people B0 missed on Singapore; B5 ran out of GPU
# memory next to DA3. Any Cityscapes SegFormer (b0-b5) works per run.
MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
MOVERS = ("person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
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

_models = {}


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_segmenter(model_id=None):
    """(processor, model, {class name: id}) for a Cityscapes SegFormer
    (default MODEL_ID): the default is built at startup on a Space (see
    streetview_to_3d.gpu), anything else on first use."""
    model_id = model_id or MODEL_ID
    if model_id not in _models:
        from transformers import AutoModelForSemanticSegmentation, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForSemanticSegmentation.from_pretrained(model_id).to(_device()).eval()
        label_ids = {name: int(i) for i, name in model.config.id2label.items()}
        _models[model_id] = (processor, model, label_ids)
    return _models[model_id]


def drop_movers(paths, model_id=None, classes=None):
    """One boolean mask per image path, True on cars, people, poles and the
    like. model_id: another Cityscapes SegFormer (see get_segmenter);
    classes: the class names to drop instead of DROP."""
    import torch
    processor, model, label_ids = get_segmenter(model_id)
    unknown = set(classes or ()) - set(label_ids)
    if unknown:
        raise ValueError(f"not Cityscapes classes: {sorted(unknown)}")
    drop = [label_ids[n] for n in (classes or DROP)]
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
