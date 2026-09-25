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

### Explore and edit a reconstruction

Open `streetview_to_3d/visualise/viewer.html` directly in Chrome. It is a packaged viewer, with
no server needed; Three.js still loads from a pinned CDN version.
Use **Open scene** for a folder containing `scene.json` and its PLYs, or
**Open PLY / files** for a single cloud or a multi-file selection. Folder drops
also work. The Gradio viewer is built from the same sources.

The shared top bar switches between Inspect and Fly:

- **Inspect**: orbit, pan and zoom; click a piece to select it, double-click to
  focus.
- **Fly**: mouse to steer, WASD to move, Q/E down/up and Shift to boost.
  Escape returns to Inspect and releases the mouse. Selection is retained.
- **Scene Manager** (local editor only): expand pieces to see their nodes.
  Select a piece to move it as a group, or a node to adjust just that cloud.
  Selection enables Move/Rotate handles; Inspect pauses editing. Escape cancels an active drag. Numeric offsets are east/north/up in
  metres and heading in degrees. Undo/Redo supports Ctrl/Cmd+Z and Shift+Z.

Selection, hide/show choices, tool choice and history survive mode switches.
**Isolate** temporarily hides other pieces; **Show all** clears both
isolation and explicit hiding. The confidence control under **Connection confidence**
changes connected-component selection groups, never placement. Regrouping
keeps hidden-node choices and follows the previously selected node into its
new group. Individual node selections remain individual when regrouping. The rules match `Scene.pieces(min_confidence)`.

Saved node transforms place the clouds in world coordinates. A single raw
component can be previewed without transforms; Edit's **Place from GPS** gives
it a starting placement with fixed scale (default 1.3 metres per DA3 unit).
This does not align road height. Separate/partially placed components require
`python -m streetview_to_3d.postprocess.pipeline --dir <scene-folder>` first.

**Reset** returns to its opening placement, or the prepared GPS baseline.
**Download scene.json** requests a new JSON download, without changing PLYs or
silently overwriting your files. Replace the JSON beside the original PLYs to
persist edits. Hidden nodes are saved too. Automatic alignment can overwrite
manual placements; `postprocess.render_pieces` uses saved placements directly.

### Viewer development

Edit `streetview_to_3d/visualise/viewer_src/`, not the generated `streetview_to_3d/visualise/viewer.html`:

- `template.html`, `styles.css`, `ui.js`: one shared viewer shell, toolbar,
  footer and view-settings panel for all hosts.
- `scene-manager.html`, `scene-manager.js`: optional left sidebar with the
  piece/node hierarchy, transform controls, history and save actions.
- `state.js`: selection, visibility and mode state.
- `scene-store.js`, `scene-format.js`, `files.js`: loading, transforms, history,
  export and compatibility with `scene.py`.
- `navigation.js`, `bird.js`: camera input and bird flight.
- `editor.js`: transform-handle transactions; `viewport.js`: rendering/picking.
- `app.js`: wiring and user actions.

Build the portable HTML with Python's standard library:

```bash
python -m streetview_to_3d.visualise.build_viewer
python -m streetview_to_3d.visualise.build_viewer --check
```

Gradio builds from these sources automatically in viewing-only mode (orbit and flight).
The local `viewer.html` retains the editing tools. Copy that HTML wherever you need it. Development tests are optional
Node tooling, not runtime/build requirements:

```bash
npm ci --prefix streetview_to_3d/visualise
npm test --prefix streetview_to_3d/visualise
npm run format --prefix streetview_to_3d/visualise
python -m unittest discover -s streetview_to_3d/visualise/tests -p 'test_*.py'
```

The DOM integration tests use real Three.js math/controls with GPU drawing
stubbed; they do not substitute for visual browser QA.

### Coordinate convention

Everything in this repo is **Y-DOWN**: `+Y` points at the ground, not the sky.

That is DA3's own computer-vision convention and it is never changed on the
way through -- `load_pieces` scales height but does not flip it, and every
transform downstream inherits it. `streetview_to_3d/visualise/viewer.html` applies
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
the clicked nodes                    Street View nodes, and their real links
  ↓  ask Google and Apple what imagery is near each one (metadata only)
date graphs                          one isolated graph per capture date
  ↓  download, then walk + join in one GPU call (reconstruct/)
scene.json + one .ply per node       DA3 geometry, and each camera's DA3 position beside its pano's lat/lon
  ↓  postprocess.gps_fit             each piece fitted to its own GPS
  ↓  postprocess.road_align          onto its road, across it, then height and tilt
scene.json, now with a transform per node   its stored .ply -> world metres
```

Solving is slow and the answer is small, so it is saved rather than baked
into a merged cloud. `postprocess.render_pieces` builds one from it
without solving. For each stage in detail, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Placement

`python -m streetview_to_3d.postprocess.pipeline --dir <run dir>` places a scene; add
`--merge out.ply` to also write it as one cloud. The Space runs the same
call right after reconstructing.

1. **GPS** (`gps_fit/load_pieces.py`). A piece is the nodes sharing one DA3
   frame. Its cameras are fitted to their panoramas' GPS: a rotation and an
   offset per piece, but ONE scale for the whole scene -- the median over
   pieces whose cameras span at least 8 m, or `config.DA3_UNITS_TO_METRES`
   when none do. A scale per piece would turn GPS noise into pieces of
   different sizes. A lone panorama has nothing to fit a heading from, so
   it is turned by its own camera heading.
2. **Road line** (`road_frames.py`, `node_center_to_road_line.py`). The
   street graph is chained into roads, and each piece's cameras are seated
   on its road's line. A piece of 3+ nodes only slides; a 2-node piece may
   also turn.
3. **Across the road** (`cross_road.py`). The line follows whichever lane
   the car drove, so pieces on the same road are compared by the height
   profile of the ground across it, and every sideways shift (plus a turn
   of at most 5 degrees) is solved at once.
4. **Height** (`ground_elevation.py`). Google's per-node elevation gives the
   real ground along each road; each piece gets one height offset and one
   tilt onto it, measured from the ground under its own camera track.

Nothing corrects a piece ALONG its road: a straight road looks the same at
every point along it, so GPS keeps that.

## Dev notes

**Solo-score vs. pairwise DA3 experiment** (2026-08-19, real data; the one-off script was removed once its numbers were recorded here):

- Hypothesis confirmed: a candidate's solo DA3 self-consistency score predicts pairwise success likelihood.
  - min-score 6 → 33% pairwise success
  - min-score 8 → 67%
  - min-score 11 → 67%
  - min-score 13+ → 100%
- DA3 model load: **8.93s**
- Solo-score call: avg **1.36s**
- Pairwise call: avg **1.99s**
- First calibration of `SECONDS_PER_DOT_ESTIMATE` in `reconstruct/walk_graph.py` (since raised from real runs' `timing:` logs).

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
