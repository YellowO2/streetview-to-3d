# Procedural streets and buildings

The detailed facade generator reuses the pure Python geometry core from
[p-schulz/osm_building_grammar](https://github.com/p-schulz/osm_building_grammar),
pinned at `81f1b50eae048645b2e620056a7cb2c84709bcfa`. Its unmodified `grammar.py`
and `config.py`, Apache-2.0 license and provenance are under `_vendor/`.
No Blender installation, texture downloads or browser dependencies are required.
Our adapter and style settings live in `facade_geometry.py`.

## Physical architecture

Ordinary buildings with point spacing at most 1.6 m receive geometry selected from
OSM building type, material, construction date, architecture, footprint and storeys:

- Historic masonry: tall windows, deep raised surrounds, pilasters and projecting
  floor bands/cornices. Office facades use wider/taller glazing and slab bands.
- Apartments: alternate bays of 1.15 m projecting balcony slabs with three-sided
  guards, window surrounds and floor bands. Houses and warehouses use different
  window dimensions and omit balconies.
- Ground entrances: thick surrounds and 1.2 m canopies on the inferred street side.
- Roof silhouettes: a small chimney on pitched roofs or a housing on flat roofs,
  wholly within the mapped roof; stacked building parts omit this equipment.

Frame fronts are closed back to form physical sides; glass sits behind the raised
surround. These are deep window assemblies, **not holes cut into the original wall**.
Mapped building mass, terrain settlement and existing roof geometry remain in charge.
Generic building parts inherit type/style hints from their containing outline so,
for example, a church's untyped parts do not acquire residential balcony rules.

Both the near-point and distant-mesh paths use the same geometry generator.
Physical facades do not receive the previous painted window grid. Geometry and
colours are deterministic. Nearby walls fitted to DA3 omit guessed architecture;
shared walls omit projecting facade modules. Detail is omitted on very short edges,
very large footprints, utility structures and landmarks with incompatible types.
Farther buildings keep the cheaper facade-colour treatment and roof trim.

All added architecture is plausible inferred geometry, not surveyed facade detail.
Sparse points can still hide narrow mullions; metre-scale balcony/entrance volumes
and continuous bands are the intended visual cues. The density threshold controls
which buildings receive extra geometry and is based on the saved point spacing,
not the moving viewer camera. Existing exported scenes must be rebuilt.

## Roads and pavements

- Explicit sidewalk left/right/both tags and side-specific widths produce pavement
  ribbons. `separate` requires a mapped `highway=footway, footway=sidewalk` way.
- Pavements have a 14 cm rise, pale edge ribbons and vertical curb faces. Generated
  ribbons avoid mapped building footprints and yield to carriageways at junctions.
- Explicit integer lane counts (2–8) produce illustrative dashed lane dividers on
  paved roads. `lane_markings=no` suppresses them; junctions clear paint for 8 m.
- Surfaces have small deterministic colour variation; crossings and street furniture
  use the geometry described below.

Rebuild through `terrain.build(scene_dir)` to obtain new geometry. Cached panorama
painting and DA3 coverage can replace procedural colours close to cameras.

## Street geometry pass

Roads now export the same street furniture to near point clouds and far
triangle meshes. Tagged pavements have raised kerbs and 1.2 m paving panels
separated by actual dark polygon seams. These are geometry, rather than a
texture or vertex colour approximation. Existing lane dividers remain.
Explicit marked crossing nodes add 3.6 m wide zebra crossings aligned to the
nearest road; unmarked crossings and explicit marking opt-outs stay unpainted.

The OSM request also includes crossing, traffic signal, street lamp, bench and
waste basket nodes. Mapped furniture produces simple solid geometry: lamp
columns and arms, benches with slats and supports, bins, and static traffic
signal housings. A traffic control node on the carriageway is moved to the
nearest road's side because the node often represents a junction control,
not the physical pole. Signal lenses do not simulate an active traffic state.
Lamps may additionally be inferred at a 28 m rhythm on explicitly lit,
tagged pavements; mapped lamps suppress nearby inferred copies. Other street
furniture is never inferred from an untagged road.

This is procedural interpretation, not an exact survey. Pavement panel layout
is a regular world-aligned grid; the OSM sidewalk width and footprint determine
its boundary. Fine seams and small fittings remain subject to point spacing.
The normal build keeps 80 cm at 100 m and 1.5 m at 200 m from the reconstructed
edge, easing into coarser spacing between 250 and 400 m. These defaults also
control which buildings receive physical facades.


## Normal workflow

New scenes receive these details automatically through the existing scene build
and `terrain.build(scene_dir)`. The viewer uses the existing `buildings`, `blocks`,
`roads` and `terrain` scene assets; no preview-specific loader or manual patch is
needed. To update an already exported scene, from the repository root run:

```bash
python -m streetview_to_3d.postprocess.terrain /path/to/scene_directory
```

The expanded OSM query refreshes an older cache once to include tagged street
nodes, then reuses it. Existing PLY meshes without facade layout metadata remain
supported by the viewer. Window glass is blue in both physical and coarse facades.
After editing viewer source, regenerate the packaged viewer with:

```bash
python -m streetview_to_3d.visualise.build_viewer
```
