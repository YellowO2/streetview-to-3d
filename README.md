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
python app.py

#viewer
cd streetview_to_3d/visualise && python3 -m http.server 8000
```

## Tests

```bash
pip install pytest && python -m pytest streetview_to_3d
cd streetview_to_3d/visualise && npm install && npm test
```

## Acknowledgments

- [osm_building_grammar](https://github.com/p-schulz/osm_building_grammar), vendored facade geometry core (Apache 2.0)
- [Depth-Anything-3](https://github.com/ByteDance-Seed/Depth-Anything-3) (Apache 2.0)
- [streetlevel](https://github.com/sk-zk/streetlevel), for Street View coverage and downloads

## License

MIT.
