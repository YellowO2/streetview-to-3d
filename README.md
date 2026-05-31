---
title: Street View To 3dgs
emoji: 🌖
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.15.2
python_version: '3.12'
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Turns a Google Street View location into a 3DGS scene
---

# Street View to 3DGS

Convert Google Street View panoramas into 3D Gaussian Splat scenes.
Check out demo at <https://huggingface.co/spaces/potato-bug/street-view-to-3dgs>

## Run locally

Requires an NVIDIA GPU with recent drivers and Python 3.12.

```bash
# 1. Install torch + torchvision matching your CUDA driver. Pick the right wheel index for your CUDA version. Check with `nvidia-smi`. For example:
#      CUDA 12.1 → https://download.pytorch.org/whl/cu121
#      CUDA 12.4 → https://download.pytorch.org/whl/cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Install the rest.
pip install -r requirements.txt

# 3. Run.
python app.py
```

Models (Sharp + DA3) are downloaded from the Hugging Face Hub on first run and cached under `~/.cache/huggingface/`.

## License

CC-BY-NC-4.0 (inherited from Depth-Anything-3, which is non-commercial).
