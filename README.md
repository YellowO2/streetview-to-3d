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

Convert Google Street View panoramas into 3D Gaussian Splat scenes, built on top of [panoramic-to-3dgs](https://github.com/YellowO2/panoramic-to-3dgs).
Check out the demo on [Hugging Face](https://huggingface.co/spaces/potato-bug/street-view-to-3dgs).

<table>
<tr>
<td align="center"><sub>Demo Video</sub></td>
<td align="center"><sub>Comparison with HunyuanWorld 2.0 + World Marble 1.1</sub></td>
</tr>
<tr>
<td width="50%"><a href="https://youtu.be/mzIDZWxv4vA"><img src="https://img.youtube.com/vi/mzIDZWxv4vA/hqdefault.jpg" alt="Demo video"></a></td>
<td width="50%"><a href="https://youtu.be/fYANbQXMZ_0"><img src="https://img.youtube.com/vi/fYANbQXMZ_0/maxresdefault.jpg" alt="Comparison with HunyuanWorld 2.0 + World Marble 1.1"></a></td>
</tr>
</table>

## Run locally

Requires an NVIDIA GPU with recent drivers. Python **3.12** is recommended.

```bash
# 1. Create venv and activate
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Install torch + torchvision matching your CUDA driver. 
# Pick the right wheel index for your CUDA version. 
# Check with `nvidia-smi`. For example:
#      CUDA 12.1 → https://download.pytorch.org/whl/cu121
#      CUDA 12.4 → https://download.pytorch.org/whl/cu124
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install the rest
pip install -r requirements.txt

# 4. Run
python app.py
```

Models (Sharp, DA3, and FLUX) are downloaded from the Hugging Face Hub on first run and cached under `~/.cache/huggingface/`.

## Dev notes

**Solo-score vs. pairwise DA3 experiment** (2026-08-19, real data, see `tests/debug_solo_score_experiment.py`):

- Hypothesis confirmed: a candidate's solo DA3 self-consistency score predicts pairwise success likelihood.
  - min-score 6 → 33% pairwise success
  - min-score 8 → 67%
  - min-score 11 → 67%
  - min-score 13+ → 100%
- DA3 model load: **8.93s**
- Solo-score call: avg **1.36s**
- Pairwise call: avg **1.99s**
- Used to calibrate `SECONDS_PER_DOT_ESTIMATE = 6.0` in `street_builder/reconstruction/walk_graph.py`.

## Planned

**Car/people removal pass** (not started): a later cleanup pipeline that
re-does the reconstruction using cleaned-up source images instead of the
raw panoramas.

- After a reconstruction finishes, also export a small JSON of the basic
  metadata needed to re-fetch every pano actually used (source, id/key,
  lat/lon, date) -- enough to reconstruct the same corridor's input set
  without re-running Prepare/Run.
- A separate later pass: fetch those panos, clean them (remove cars/
  people), then re-run just the join/reconstruction step
  (`street_builder/reconstruction/join_segments.py`) against the cleaned
  images.
- Possible head start: this repo already has a Flux-based object-removal
  editor (`editors/flux_editor.py`, `flux-2-klein-4B-object-remove-lora`
  in Acknowledgments below) wired up for other editing use -- worth
  checking whether it's directly reusable for this instead of building
  a new cleanup step from scratch.

## Acknowledgments

This project relies on:

- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache 2.0)
- [Apple ml-sharp](https://github.com/apple/ml-sharp) (Apple sample code license)
- [FLUX.2-klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) (Black Forest Labs)
- [flux-2-klein-4B-object-remove-lora](https://huggingface.co/fal/flux-2-klein-4B-object-remove-lora) (fal)

## License

MIT.
