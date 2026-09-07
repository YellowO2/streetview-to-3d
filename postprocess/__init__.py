"""Alignment: turning separately-reconstructed pieces into one
consistent 3D scene.

    DA3 alignment -> GPS alignment -> road alignment

- `gps`  -- the shared GPS-fitting math, and the single global
            coordinate origin every script must use.
- `road` -- road alignment, the refinement pass that removes the seams
            GPS leaves behind.
- `viz`  -- the standard interactive viewer for alignment results.

These are the real, reusable parts of the alignment pipeline. `tests/`
holds one-off experiments and checks that import from here.
"""
