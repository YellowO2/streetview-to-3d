# Viewer styles

The production viewer owns these effects. The earlier experiments/style-lab
is retained as a reference, not imported at runtime.

- presets.js: Original, Soft paint and Dither defaults.
- ui.js: toolbar selector and settings; remembers each preset during the page session.
- controller.js: lazy postprocessing, resize, point-motion lifecycle and editor overlay rendering.
- anime.js, dither.js: colour passes; one is active at a time.
- points.js: patches existing point materials, leaving geometry and scene.json unchanged.
- environment.js: camera-centred sky dome. Terrain is loaded from the scene.

Soft paint is the default. Photoreal (`original`) uses the direct render path. Builders can set
config.style to original, paint or dither. Effects are recursively embedded
by build_viewer.py, so standalone and Gradio viewers share the implementation.

The environment has about 3,500 triangles and uses ordinary rasterization. It is
stationary in world space, so both Inspect and Fly respond naturally. The sky follows
camera translation but retains its world orientation. Environment can be disabled.
No volumetric clouds, texture downloads or extra dependencies are added.

Postprocessing uses two half-float render targets plus depth textures at the
viewer's existing pixel ratio cap. Original avoids the extra passes. Editor
handles/highlights are drawn after processing, preserving axis colours.
Point motion and reveal stop while editing; point positions and saved transforms
are never changed by an effect.

Spark 0.1.10 splats work through the same colour pipeline. Their transparent
rendering does not provide reliable surface depth, so the point-depth mask
is disabled for splats. Point floating,
scan and reveal controls are hidden for splats. A future Spark-specific modifier
can add these without changing the point adapter.

Validation: existing scene/edit/history/load tests; added preset persistence,
environment anchoring/disposal, and geometry-preservation tests. Browser checks
cover the real point capture, editing/undo with ungraded handles, and a synthetic
81-Gaussian PLY loaded through Spark. The in-app browser blocks pointer lock;
Fly returns safely to Inspect there. Test actual flying in a regular browser.

Soft paint combines the colour pass with circular points, a 1.2× size multiplier,
and 90% default point density. Legacy anime/painterly config names map to paint.
View settings has a 10–100% density control remembered per style. A deterministic
position hash clips rejected points in the vertex shader, reducing rasterization
without CPU resampling, geometry mutation or camera-dependent flicker. It does
not reduce vertex processing or memory. Density is unavailable for Gaussian
splats. Point size remains a user multiplier on the preset size.

Dither grain has a minimum 3 CSS pixel cell size and a maximum of 480 cells
along the viewport’s longest edge. This caps image-pattern density, independently
of scene point count and display pixel ratio; it does not decimate geometry.

water.js draws a placed scene's water (water.json, postprocess/water.py): each
body one flat shape at its own level, with animated normals, analytic blue-sky
reflection and a sun glint -- no reflection of the scene, no waves or foam. One
shared material; its motion pauses while editing.
