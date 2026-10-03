# Viewer effects

Styles: Soft paint (default), Dither, Characters (Soft paint with points drawn as characters) and
Photoreal (`original`, direct render, no passes). Effects never change point positions or scene.json.

- presets.js, ui.js: style defaults and the toolbar/settings controls.
- controller.js: per-frame style uniforms, demos, motion ticks and the postprocess chain.
- anime.js, dither.js: colour passes (one active at a time); splats skip the depth mask.
- environment.js, clouds.js: camera-centred sky dome and paint-style point clouds.
- points.js: patches DA3 and map point materials (float, scan, density, wind sway).
- world-points.js: GLSL hooks every world-point shader shares, built from demo.js (rise/swirl),
  shot.js (gun holes), glyphs.js (characters) and thin.js (far-off thinning).
- blocks.js: buildings drawn as lit dabs; land.js, moving.js reuse it for land and life.
- scatter.js: roads.ply triangles scattered with points; haze.js: distance haze for dabs.
- water.js: water points mixing their colour with a Reflector mirror, waves and sun glint.
- traffic.js, birds.js, boats.js, ducks.js, cats.js: moving things from life.json.
- tune-panel.js: `?tune` sliders for tuned constants; util.js: shared GLSL and math helpers.

All modules are embedded into viewer.html by build_viewer.py.
