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
short_description: Reconstructs a street corridor into a 3D point cloud
---

# Street Builder

Reconstructs a walkable street corridor into a joined 3D point cloud from Google Street View / Apple Look Around panoramas, built on top of [panoramic-da3](https://github.com/YellowO2/panoramic-da3)'s DA3 core.
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

## Layout

The pipeline splits by what a stage needs to run:

| | |
|---|---|
| `services/` | fetching panoramas, DA3, the GPU runner |
| `street_builder/` | **GPU stage.** Panoramas in, chunks of `.ply` + metadata out |
| `postprocess/gps_fit/` | **CPU stage 1.** Which nodes form a piece, and where that piece sits in metres |
| `postprocess/road_align/` | **CPU stage 2.** Correcting the heading, position and height GPS leaves wrong |
| `postprocess/` | what both produce: `piece_transforms.json`, and rendering from it |
| `visualise/` | the graph page, the point-cloud viewer, the segment selector |
| `tools/` | one-off drivers, experiments and conversions. Nothing imports these |
| `data/` | committed inputs |
| `images/` `splats/` `build/` `ntu/` | run outputs and fetched data, all ignored |

`splats/` is where each reconstruction run writes; it is created on import by
`paths.py` and is empty until something runs.

### How a reconstruction becomes an aligned scene

```
google_graph.json                    every Google panorama in the area, and how they link
  ↓  ask Google and Apple what imagery is at each spot
downsampled_and_fetched_graph.json   corridor resampled to dots + a pano census (metadata only)
  ↓  pick dates, pick a corridor
graph.json                           the nodes actually reconstructed
  ↓  street_builder (GPU) — uploads to HuggingFace, not local
piece_*.ply + piece_*_meta.json      DA3 geometry, and each camera's DA3 position beside its lat/lon
  ↓  postprocess.gps_fit             that pairing is what makes the fit possible
  ↓  postprocess.road_align          heading, position, then height and tilt
piece_transforms.json                one 4x4 per piece: stored .ply -> world metres
```

Solving is slow and the answer is small, so it is saved rather than baked
into a merged cloud. `postprocess.render_pieces` builds any subset from it
without solving.

Full NTU is not stored locally — it lives on HuggingFace under
`cli_raw/<chunk_id>/`. Alignment currently assumes the pieces form ONE
corridor; a campus is a network, so aligning all of NTU needs corridors
split at junctions first.

## Alignment

Pieces are brought into one consistent scene in three stages, in `alignment/`:

```
DA3 alignment  ->  GPS alignment  ->  road alignment
```

GPS gets every piece roughly right but leaves visible seams: pieces sit
at slightly different heights and slightly off sideways, and a piece
built from a single panorama can face almost any direction (one GPS
point pins a position but says nothing about a heading).

**Road alignment** (`alignment/road.py`, run over a set of pieces with
`python -m alignment.run_road_align --dir <pieces> --out out.ply`)
closes those seams using the road surface itself:

1. **Look straight down at each piece.** For every ground cell keep the
   colour of its highest point — a plain top-down photo, no filtering.
2. **Grey cells are road.** That gives each piece's road as a 2D shape.
3. **Measure which way that road runs.** A road is long and thin, so its
   direction is the angle at which it is narrowest measured across. The
   painted white line is fitted separately as an independent check.
4. **Decide whether the heading may be touched**, by comparing the road
   direction against the piece's *own* GPS camera track:
   - they agree → GPS already got it right, leave it alone
   - they disagree → the heading is wrong, solve for it
   - no track at all (single-node piece) → nothing ever constrained the
     heading, solve for it
5. **Slide sideways across the road** until the two road shapes overlap.
   The kerbs make this sharp.
6. **Shift up or down** until the two road surfaces sit at the same height.
7. **Leave the along-road direction alone.** A straight road looks
   identical at every point along its own length, so nothing in the
   imagery can determine it — GPS keeps that one, permanently.

Three things this design exists to avoid, each found by measurement:

- **Neighbouring pieces legitimately differ in heading.** Where the road
  curves, two pieces can be 9° apart while each matches its own GPS
  track to within 1°. Treating that as error drags pieces ~10 m off GPS,
  hence the step-4 self-check rather than a node-count rule.
- **Overlap area cannot determine rotation.** On a straight road it
  varies by ~0.03 IoU across ±20°. Headings are solved by matching road
  *directions* (sharp to a few degrees); overlap is used only to pick
  between the two 180°-opposed choices.
- **Pieces are aligned to their single best-overlapping neighbour**, not
  to everything placed so far — against the union, a piece with a small
  genuine overlap slides sideways onto some other road entirely. A
  correction larger than a few times the piece's own GPS residual is
  rejected as exactly that failure.

A panorama's blind spot leaves a hole in the middle of its own road;
every step above is written to tolerate it.

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
- [streetlevel](https://streetlevel.readthedocs.io/en/master/streetlevel.lookaround.html) (`streetlevel.lookaround`) -- fetching Apple Look Around panorama coverage/metadata and downloading panorama faces (`get_coverage_tile`/`get_coverage_tile_by_latlon`, `get_panorama_face`/`download_panorama_face`).

## License

MIT.
