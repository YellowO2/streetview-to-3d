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

For a stage-by-stage description of how the pipeline works, see
[ARCHITECTURE.md](ARCHITECTURE.md).

### Coordinate convention

Everything in this repo is **Y-DOWN**: `+Y` points at the ground, not the sky.

That is DA3's own computer-vision convention and it is never changed on the
way through -- `load_pieces` scales height but does not flip it, and every
transform downstream inherits it. `Documents/viewer.html` applies
`geometry.rotateX(PI)` at display time, which is the only place the flip
happens.

| axis | direction |
|---|---|
| `+X` | east |
| `+Y` | **down** |
| `+Z` | north |

X and Z are metres in the shared `GLOBAL_ORIGIN` frame (`gps_fit.fit.real_en`),
so piece clouds, camera positions and the road polylines from `corridors` all
compare directly.

**Anything brought in from outside must be converted.** Google's per-panorama
`elevation` is metres above sea level -- Y-UP -- so it has to be negated
before it can be compared with our heights. Using it raw turns a hill into a
hole, and a vertical fit against it then squeezes the real relief out of the
scene while every individual number still looks plausible.

### How a reconstruction becomes an aligned scene

```
google_graph.json                    every Google panorama in the area, and how they link
  ↓  ask Google and Apple what imagery is at each spot
downsampled_and_fetched_graph.json   corridor resampled to dots + a pano census (metadata only)
  ↓  pick dates, pick a corridor
graph.json                           the nodes actually reconstructed
  ↓  reconstruct (GPU) — uploads to HuggingFace, not local
piece_*.ply + piece_*_meta.json      DA3 geometry, and each camera's DA3 position beside its lat/lon
  ↓  postprocess.gps_fit             that pairing is what makes the fit possible
  ↓  postprocess.road_align          heading, position, then height and tilt
piece_transforms.json                one 4x4 per piece: stored .ply -> world metres
```

Solving is slow and the answer is small, so it is saved rather than baked
into a merged cloud. `postprocess.render_pieces` builds any subset from it
without solving.


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

## Checking a whole reconstruction against GPS

How well a merged reconstruction matches reality, and the DA3-to-metres
scale, are both measured the same way -- off node metadata alone, no point
clouds:

```bash
python -m tools.validate_gps_alignment --group g_L16_0 --out /tmp/fit.json
python -m postprocess.gps_fit.discover_pieces --in /tmp/fit.json --out /tmp/islands.json
python -m visualise.graph_page --in /tmp/islands.json --group-field island \
    --pos-field fitted_en --ref-field real_en --out /tmp/islands.html
```

`discover_pieces` cuts each chunk where its own nodes stop fitting their
GPS, then merges adjacent pieces back wherever the combined fit still
holds. On NTU's 117 chunks that gives 52 islands fitting to 3.5 m median,
against 330 m for one transform across the whole tree.

Note the pipeline itself uses only the cutting half (`resolve`, via
`postprocess/gps_fit/split_by_fit.py`). The merge-back is CLI-only.

### The DA3-to-metres constant

`config.DA3_UNITS_TO_METRES` is measured, not guessed. To re-measure it:

```bash
python -m tools.measure_da3_scale --group g_L16_0 --sweep
```

It cuts each chunk where its own nodes stop matching their GPS, fits every
surviving piece of 4+ nodes alone, and prints the spread per cut. Take the
value where the mean meets the median -- that is where nothing is left
skewing the sample. On NTU it holds at 1.33-1.35 across cuts from 12 m
down to 0.75 m.

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
- Used to calibrate `SECONDS_PER_DOT_ESTIMATE = 6.0` in `reconstruct/walk_graph.py`.

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
  (`reconstruct/join_segments.py`) against the cleaned
  images.

## Acknowledgments

This project relies on:

- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache 2.0)
- [streetlevel](https://streetlevel.readthedocs.io/en/master/streetlevel.lookaround.html) (`streetlevel.lookaround`) -- fetching Apple Look Around panorama coverage/metadata and downloading panorama faces (`get_coverage_tile`/`get_coverage_tile_by_latlon`, `get_panorama_face`/`download_panorama_face`).

## License

MIT.
