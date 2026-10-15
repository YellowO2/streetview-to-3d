---
title: Street View To 3D
emoji: 🌆
colorFrom: pink
colorTo: indigo
sdk: gradio
sdk_version: 6.15.2
python_version: '3.12'
app_file: main.py
pinned: false
license: mit
short_description: Reconstructs a street corridor into a 3D point cloud
---

# Street View to 3D

Reconstructs a stretch of street as 3D point cloud from Google Street View panoramas. Try it on the [Hugging Face Space](https://huggingface.co/spaces/potato-bug/street-view-to-3d).

https://github.com/user-attachments/assets/db1c7f82-ee53-4376-a78a-ec393c426604

For a single panorama as a Gaussian splat, see [panoramic-to-3dgs](https://github.com/YellowO2/panoramic-to-3dgs).

## Run locally

Requires an NVIDIA GPU and Python 3.12.


```bash
python3.12 -m venv .venv && source .venv/bin/activate
# torch + torchvision for your CUDA version (see `nvidia-smi`), e.g. CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python main.py

#viewer
cd streetview_to_3d/visualise && python3 -m http.server 8000
```

## Tests

```bash
pip install pytest && python -m pytest streetview_to_3d
cd streetview_to_3d/visualise && npm install && npm test
```

## Layout

```
streetview_to_3d/   the engine: two machines side by side, one per operation, and what they share
  gather_panoramas/  1. prepare (the "Prepare" button), no GPU: main.py → fetch_corridor_nodes →
                   date_graphs → downloaded; tools/ dates, corridor_points, the pano lookups
  build_scene/  2. build (the "Reconstruct" button): main.py → reconstruct → postprocess → save;
                   it makes the models and hands them in
    models/       DepthModel (DA3), Segmenter (the masker); built at startup on a Space;
                   python -m scripts.draw_mask draws the masker on one pano
    reconstruct/  GPU: main.py → open_scene → PanoDepth → walk (one GPU window) → pieces → assemble
    tools/        pano_mask (what the masker drops), for reconstruct and postprocess
    postprocess/  CPU: main.py → place, clean/, photos, fill/ (main.py), world/ (main.py); tools/ paint
                   and ground_maths;
                   python -m streetview_to_3d.build_scene.postprocess.main --dir <run> [--google] re-runs it on a saved run
      world/      the map round the scene: loaders first (OpenStreetMap, HeightMap, elevations_around,
                   WaterMap, Satellite, pano_views, driving_side), then Elevation, water_outline,
                   Water, roads/, outlines/, land/, land_colours, buildings/, car_roads, life,
                   world_files, scene_files; with Settings.google, google_points and google/
                   (the Google 3D Tiles merge); tools/ seams, osm_roads, osm_tags, styles, geometry...
  tools/        helpers the parts use inside themselves: paths (where data lives, new_run_dir),
                   geo, ply, run_async, on_gpu (the one way to a GPU), DisjointSet,
                   streetview_fetch/ (lookups and downloads); deeper tools/ folders hold their
                   own subject's
  types/        Scene (Node, Pano, Edge, Cloud)
  visualise/    4. viewer; viewer_src/ core/ flight/ style/ world/ life/ ui/;
                   python -m scripts.write_viewer rebuilds viewer.html
main.py   the whole app: the UI, handed the engine's two operations (gather_panoramas to prepare, build_scene to build)
ui/     the Gradio page; each control checks what it is given, then runs its job
  main.py       the page: map_selection → page_controls → prepare_button → reconstruct_button → pano_api
  on_prepare/   button 1's job: the engine's gather_panoramas on the traced Route
  on_reconstruct/  button 2's job: RunChoices → Settings → the engine's build_scene → the scene shown
  map_selection/  the map: main.py → map_widgets, then location_box (load_area), expand_button
                   (expand_selection), map_clicks (redraw_area for drags), radius_box,
                   clear_button; tools/ map_view...
  pano_api      the masker and DA3 on one pano, API only
  tools/        page (file_url, iframe), map_state, viewer, gallery/ (sharing: publish uploads
                   to the public gallery, index.html shows it)
scripts/  for developers: write_viewer (rebuild viewer.html), draw_mask (the masker on one pano),
                   clean (drop blobs or sheets from a saved scene); run as python -m scripts.<name>
```

## Acknowledgments

- [osm_building_grammar](https://github.com/p-schulz/osm_building_grammar), vendored facade geometry core (Apache 2.0)
- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache 2.0)
- [streetlevel](https://github.com/sk-zk/streetlevel), for Street View coverage and downloads

## License

MIT.
