"""Step 2, the GPU: panoramas in, point clouds out. main.reconstruct opens the scene
(open_scene.py), walks the panos in one GPU window, DA3 run by PanoDepth.py on each test
(walk.py grows pieces from DA3 edges that hold), shapes the pieces (pieces.py) and puts
them into the scene (assemble.py).
"""
