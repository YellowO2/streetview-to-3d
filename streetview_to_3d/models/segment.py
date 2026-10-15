"""Label each pixel of some images with the class it shows.

Cars, people, thin poles/signs and water, marked so DA3 never turns them into points.

Movers ghost when panos merge, DA3 smears thin things into streaks and lays water at street
height. Each pano is segmented once, whole (Cityscapes SegFormer, plus an ADE20K one for water);
each DA3 view cuts its part of that mask, and the class map is kept with the scene for the fill.
"""
import numpy as np
from PIL import Image

# B2 caught poles and people B0 missed on Singapore; B5 ran out of GPU
# memory next to DA3. Any Cityscapes SegFormer (b0-b5) works per run.
MODEL_ID = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
# Cityscapes' 19 classes by id (_segmenter checks), so a saved class map reads without the model
CITYSCAPES = ("road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
              "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
              "motorcycle", "bicycle")
CLASSES = CITYSCAPES + ("water",)
LABEL_IDS = {name: i for i, name in enumerate(CLASSES)}
WATER_MODEL_ID = "nvidia/segformer-b2-finetuned-ade-512-512"
WATER_CLASSES = ("water", "sea", "river", "lake", "swimming pool")   # ADE20K's
BATCH = 8   # images per pass; 16 ran a bigger model out of GPU memory
# The masker's input width, the image's own shape kept (None: the
# processor's square 512 x 512, which stretched the image)
MASK_W = 1024

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


def _segmenter(model_id=None, device=None):
    """(processor, model, {class name: id}) for a Cityscapes SegFormer
    (default MODEL_ID) on device (default: the GPU if there is one), the
    water one (WATER_MODEL_ID) loaded beside it: the default is built at
    startup on a Space (see streetview_to_3d.models.on_gpu), anything else on first
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


def segment(paths, model_id=None, device=None):
    """(one class-id map per image path -- Cityscapes', water marked over
    it -- and {class name: id}, as LABEL_IDS). model_id, device: see
    _segmenter; no paths only builds the models."""
    device = device or _device()
    processor, model, _ = _segmenter(model_id, device)
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
