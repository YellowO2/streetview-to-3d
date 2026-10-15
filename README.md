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
streetview_to_3d/   the engine
  main.py       a prepared corridor → reconstruct → postprocess → save (the "Reconstruct" button);
                   it makes the models and hands them in
  tools/        helpers the parts use inside themselves: paths (where data lives, new_run_dir),
                   geo, ply, run_async, pano_mask, on_gpu (the one way to a GPU),
                   streetview_fetch/ (lookups and downloads); deeper tools/ folders hold their
                   own subject's
  common/       Scene (Node, Pano, Edge, Cloud), DisjointSet
  models/       DepthModel (DA3), Segmenter (the masker); handed to the steps that run them;
                   python -m streetview_to_3d.models draws the masker on one pano
  panos/        1. which panoramas: main.py → fetch_corridor_nodes → date_graphs → downloaded
                   (the "Prepare" button); tools/ dates, corridor_points, the pano lookups
  reconstruct/  2. GPU: main.py → open_scene → PanoDepth → walk (one GPU window) → pieces → assemble
  postprocess/  3. main.py → place, clean/, photos, fill/ (main.py), world/ (main.py); tools/ paint
                   and ground_maths;
                   python -m streetview_to_3d.postprocess --dir <run> [--google] re-runs it on a saved run
    world/      the map round the scene: loaders first (OpenStreetMap, HeightMap, elevations_around,
                   WaterMap, Satellite, pano_views, driving_side), then Elevation, water_outline,
                   Water, roads/, outlines/, land/, land_colours, buildings/, car_roads, life,
                   world_files, scene_files; with Settings.google, google_points and google/
                   (the Google 3D Tiles merge); tools/ seams, osm_roads, osm_tags, styles, geometry...
  visualise/    4. viewer; viewer_src/ core/ flight/ style/ world/ life/ ui/;
                   python -m streetview_to_3d.visualise rebuilds viewer.html
main.py   the whole app: the UI, handed the engine's two jobs (panos to prepare, streetview_to_3d to build)
ui/     the Gradio page
  main.py       the page: map_section → page_controls → prepare_button → reconstruct_button
  map_selection/  the map and its candidates (map_section; tools/ nearby_nodes, expand_area...)
  pano_api      the masker and DA3 on one pano, API only
  tools/        page (file_url, iframe), map_state, viewer, gallery/ (sharing: publish uploads
                   to the public gallery, index.html shows it)
```

## Acknowledgments

- [osm_building_grammar](https://github.com/p-schulz/osm_building_grammar), vendored facade geometry core (Apache 2.0)
- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache 2.0)
- [streetlevel](https://github.com/sk-zk/streetlevel), for Street View coverage and downloads

## License

MIT.
