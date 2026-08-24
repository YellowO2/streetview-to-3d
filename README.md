---
title: Street View To 3D
emoji: 🌆
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.15.2
python_version: '3.12'
app_file: app.py
pinned: false
license: mit
short_description: Reconstructs a walkable street corridor into a 3D point cloud
---

# Street Builder

Reconstructs a walkable street corridor into a joined 3D point cloud from Google Street View / Apple Look Around panoramas, built on top of [panoramic-to-3dgs](https://github.com/YellowO2/panoramic-to-3dgs)'s DA3 core.
Check out the demo on [Hugging Face](https://huggingface.co/spaces/potato-bug/street-view-to-3d).

For the single-panorama SHARP/3DGS pipeline, see [streetview-to-3dgs](https://github.com/YellowO2/streetview-to-3dgs).

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

Models (DA3) are downloaded from the Hugging Face Hub on first run and cached under `~/.cache/huggingface/`.

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

## Acknowledgments

This project relies on:

- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache 2.0)

## License

MIT.
