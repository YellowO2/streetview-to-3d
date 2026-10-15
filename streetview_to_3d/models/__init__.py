"""The models every step shares: DA3 depth (da3.py), the segmenter (segment.py) and the one
way to a GPU (on_gpu.py). ui/pano_api.py runs them on one pano over the Space's API; drawn locally:

    python -m streetview_to_3d.models PANO_ID OUT.jpg
"""
