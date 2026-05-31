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
license: mit
short_description: Turns a Google Street View location into a 3DGS scene
---

# Street View to 3DGS

Convert Google Street View panoramas into 3D Gaussian Splat scenes.
Check out the demo on [Hugging Face](https://huggingface.co/spaces/potato-bug/street-view-to-3dgs).

<table>
<tr>
<td align="center"><sub>Demo Video</sub></td>
<td align="center"><sub>Comparison with HunyuanWorld 2.0 + World Marble 1.1</sub></td>
</tr>
<tr>
<td width="50%"><video src="demo.mp4" controls></video></td>
<td width="50%"><a href="https://youtu.be/fYANbQXMZ_0"><img src="https://img.youtube.com/vi/fYANbQXMZ_0/maxresdefault.jpg" alt="Comparison with HunyuanWorld 2.0 + World Marble 1.1"></a></td>
</tr>
</table>

## Run locally

Requires an NVIDIA GPU with recent drivers. Python **3.12** is recommended.

```bash
# 1. Create venv and activate
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install torch + torchvision matching your CUDA driver. Pick the right wheel index for your CUDA version. Check with `nvidia-smi`. For example:
#      CUDA 12.1 → https://download.pytorch.org/whl/cu121
#      CUDA 12.4 → https://download.pytorch.org/whl/cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install the rest
pip install -r requirements.txt

# 4. Run
python app.py
```

Models (Sharp + DA3) are downloaded from the Hugging Face Hub on first run and cached under `~/.cache/huggingface/`.

## Acknowledgments

This project relies on:

- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache 2.0)
- [Apple ml-sharp](https://github.com/apple/ml-sharp) (Apple sample code license)

## License

MIT.
