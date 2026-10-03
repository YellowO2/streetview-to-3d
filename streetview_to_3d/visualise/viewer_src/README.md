# Viewer source

build_viewer.py embeds every `.js` here as `@viewer/<path>` (e.g. `@viewer/world/blocks`) into
viewer.html, inside template.html with styles.css and ui/scene-manager.html.
Styles: Soft paint (default), Dither, Characters and Photoreal (`original`, no passes); they never
change point positions or scene.json.

- core/: app.js wires everything; viewport.js (renderer, camera, styles), scene-store.js and
  scene-format.js (loading, scene.json), files.js (dropped files), state.js (modes,
  selection, undo), editor.js (move gizmo), start-view.js, export.js (the scene as one PLY).
- flight/: navigation.js (orbit/fly), flight-motion.js, the chase bird (bird.js, bird-paint.js,
  bird-plume.js), gun.js and the holes it shoots (shot.js).
- style/: presets.js, controller.js (per-frame uniforms, demos, postprocess chain), points.js
  (patches point materials), world-points.js (shared GLSL hooks from demo.js, shot.js, glyphs.js,
  thin.js), anime.js and dither.js (colour passes), haze.js (distance haze for dabs).
- world/: blocks.js (buildings as lit dabs), land.js, scatter.js (roads.ply), water.js,
  environment.js and clouds.js (sky dome, clouds).
- life/: moving.js (dab fleets from life.json), traffic.js, flock.js (sky birds), boats.js,
  ducks.js, cats.js.
- ui/: shell.js (toolbar, panels), style-controls.js, scene-manager.js/.html (edit mode),
  tune-panel.js (`?tune` sliders).
- util.js: shared GLSL and maths helpers.
