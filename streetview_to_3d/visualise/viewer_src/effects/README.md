# Viewer styles

The production viewer owns these effects. The earlier experiments/style-lab
is retained as a reference, not imported at runtime.

- presets.js: Original, Soft paint, Dither and Voxel world defaults.
- ui.js: toolbar selector and settings; remembers each preset during the page session.
- controller.js: lazy postprocessing, resize, point-motion lifecycle and editor overlay rendering.
- anime.js, dither.js: colour passes; one is active at a time.
- points.js: patches existing point materials, leaving geometry and scene.json unchanged.
- environment.js: camera-centred sky dome and optional mist. Terrain is loaded from the scene.

Soft paint is the default. Photoreal (`original`) uses the direct render path. Builders can set
config.style to original, paint, dither or voxel. Effects are recursively embedded
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
rendering does not provide reliable surface depth, so the point-depth fog/mask
is disabled for splats. Terrain still has ordinary distance fog. Point floating,
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
splats and voxel proxies. Point size remains a user multiplier on the preset size.

Dither grain has a minimum 3 CSS pixel cell size and a maximum of 480 cells
along the viewport’s longest edge. This caps image-pattern density, independently
of scene point count and display pixel ratio; it does not decimate geometry.

Voxel world (`voxel`) is a static point-cloud preset in voxel.js. Occupied local
cells average source colours and render as lit instanced cubes. The requested
cell size is scene radius × .008 × Block size; a shared 60,000-instance budget
is divided across source clouds, growing cells as needed. Each cloud has its own
grid, so overlapping pieces are not merged. Proxy transforms and visibility
follow the source each frame; source visibility is restored even if rendering
throws. Geometry and exports are unchanged. Instances rebuild on scene load or
block-size change (on slider release), not on camera movement. Gaussian splats
and point-motion controls are unavailable for this preset. First conversion of
large scenes can briefly pause the UI; animation is deferred.

Flooded world is a shared Surroundings toggle (including Original), not another
colour preset. water.js owns a two-triangle world-horizontal opaque surface;
height is measured above the loaded bounds' minimum Y in scene-radius units.
The shader uses animated normals, analytic blue-sky reflection and a sun glint.
It does not reflect scene objects, simulate waves, or provide shoreline foam.
Water forces the blue sky while enabled. Settings stay
across style switches; source positions and exports are unchanged. Motion pauses
while editing, or with Still water (defaults on for reduced-motion users).
Spark scenes disable water because transparent splats do not reliably occlude it.
