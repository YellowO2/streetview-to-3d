This is a pipeline that allows users to select a region on a map, then obtain a 3d point cloud representation of it.
It can be seen as google map -> 3d point cloud.

The architecture is as follows:

- 1. map_selection  (ui/map_selection/)
Enables the user to select the region on the map, either by clicking nodes one by one or via a radius. BFS is used to guarenteed a connected graph.
The start_node is the closest real google node to the inputed lon-lat of user.
Output: a `selection_graph` -- the raw Google-only nodes + edges the user selected. In code this lives as `state["selected"]` (node keys) + `state["selected_edges"]` (edges).`selection_graph` is the term we use to talk about that pair together.

- 2. build_graph  (build_street_graph/)
Input: `selection_graph`

With the `selection_graph` from map_selection, we do a series of processing to build our own graph.

    - 2.1 Build the fetch_graph  (fetch_nodes.py: `corridor_points`, `fetch_corridor_nodes`)
    The fetch_graph's dots ARE the selection_graph's own real nodes -- `corridor_points` builds the dot/adjacency structure directly from the corridor's real edges, no synthetic in-between sampling. For each dot, `fetch_corridor_nodes` fetches the nearby Google panoramic metadata (no images yet; Apple is off, `USE_APPLE`: its GPS sits 1-1.6 m off Google's and its depth is poor), within `POINT_MAX_DIST_M` = 5m of that dot's own real coordinates.
    Output: a fetch_graph where each dot has a bucket of candidate panoramic metadata (no images), plus that dot's ground elevation, which the same lookup already returned -- see 4.4.

    - 2.2 Build the top N date_graphs  (build_street_graph/build_graph.py: `build_corridor_graphs`; date_ranking.py)
    The fetch_graph's candidates span multiple real-world capture dates. We split it into isolated graph per date -- e.g. if the fetch_graph's contains dates A, B and C, we get 3 separate `date_graphs`.
        - 2.2.1 Select top N date_graphs (currently N = `DATE_TOP_N` = 5, each capped at `TOP_PANOS_PER_DOT` = 3 panos per dot)
        `rank_dates` scores every candidate date by coverage span (earliest-to-latest dot it reaches), then total dot count, then recency as a tiebreaker. `date_connects` then filters to dates that can structurally reach from the start zone toward a goal (walking dot-to-dot through the fetch_graph's adjacency, ONLY through non-empty directly-adjacent dots -- no flood past an empty one) -- a date only counts toward the top N if it passes this reachability check, not just the ranking.
        - 2.2.2 Fetch panoramic images for the top N date_graphs
        We only download images for the top N date_graphs, to avoid fetching images for dates we'll never use.
    Output: N date_graphs, each with metadata + downloaded pano image per dot.

Output: the output of 2.2 (the top N date_graphs) feeds into step 3.

- 3. 3d reconstruction  (reconstruct/)
Input: the N date_graphs from step 2.
Turns downloaded panos into an actual 3d point cloud, via real pairwise DA3 tests between panos.

    - 3.1 Corridor pathfind (walk_graph.py: `run_pathfind_reconstruction`)
    Per date_graph, independently walk and grow the dots in a bfs fashion. Two concepts drive this: `visited` dict (has this dot been rated/visited yet) and `covered` (is this dot's location already within `point_cover_tolerance_m`=15m of some visited dot. Currently computed via `covered_points()`. A dot can be covered without itself being visited).

        - 3.1.0 Sample dates first (`_sample_dates`). A few panos per date are rated solo, spread along the corridor; a date is scored by median keep rate × coverage, dates under `MIN_KEEP_RATE` are dropped, and the rest are walked best first. The best date walks the whole corridor; each later date walks only what is still weak (every dot outside a piece of `GOOD_PIECE_DOTS`+ dots, plus a dot of overlap -- `_patch_dots`). Panos of different dates are never linked in the walk.
        - 3.1.1 Pick a seed (`pick_seed`) from THIS date_graph's closest dot to `start_node` in step 1. Only called when the BFS queue is empty but the corridor still isn't fully covered or visited (normal in-progress BFS traversal, popping the queue, is a separate branch that never calls this). Every later seed: the not-yet-visited dot closest to a still uncovered region. curr_node = start_node.
        - 3.1.2 Rate a node. When curr_node not in visited, we run `rate_node`, which does DA3 scoring on every one of node B candidate panos, discard failed ones, but always keep at least one single best-scoring candidate (even if failed) + its solo DA3 point cloud. Then mark the dot `visited`.
        - 3.1.3 Connecting to a neighbour. Set curr_node to direct structural neighbor (typical bfs with queue), rate_node it too if not yet visited, then run a real DA3 `test_edge` between the two dots' best candidates. On success, `test_edge` itself returns a jointly-computed point cloud for the pair (higher quality then solo pieces via 'rate_node'), and both nodes use their share of it: a linked node never keeps its solo cloud. If a node already holds a link's points (from an earlier successful connection), those stay, so only the NEW NODE's share is joined onto that existing piece.
        - 3.1.4 If 3.1.3 fails: no flood/skip fallback. A dot is a real selection-graph node, so a failed or empty direct structural neighbor is a genuine dead end for that date -- the neighbor is simply left unconfirmed.
        - 3.1.5 On any merge success, the neighbor is queued so its own onward neighbors get tried next.
        - 3.1.6 Repeat 3.1.1-3.1.5 until every dot is covered or visited, or `MAX_FAILED_DOTS_IN_A_ROW` dots in a row fail to link; then the next date patches what is still weak (3.1.0).
        - 3.1.8 `set_cover`: across all dates' pieces, greedily pick the fewest that cover the most of the corridor.

    Output: **segments** -- `[(clouds, path_edges, date, reached_all, node_positions, frame_poses), ...]`, one per chosen piece. `clouds` is `{node key: (points, colors)}`: DA3 reconstructs at most two panoramas at a time and a node's points enter exactly once, so every point belongs to a known node. Usually more than one if no single date's coverage alone spans the whole corridor.
    
    Pieces the walk leaves separate stay separate (reconstruct/pieces.py): each is placed by its own nodes' GPS in step 4. A second DA3 search that tried to join them ("bridges", across dates too) was removed: on real runs it almost never joined anything, and cost GPU time.
    Output: **pieces** -- `[(clouds, metadata), ...]`, one per piece. metadata is `{node_key: {lat, lon, date, position, rotation, n_views_kept, n_views_total, links}}` -- `links` being this node's DA3-confirmed neighbours and the keep counts from each joint test.

    Each piece is recorded in the run's `scene.json` as it is saved (see `scene.py`), which is what step 4 reads.


    - 3.2 Solo mode (solo.py) -- the "Link panoramas" toggle off. No walk: every Google pano is reconstructed on its own, one piece each. The panos are each route dot's Google candidates (best-ranked date within 5 years of the others) plus Google's official neighbours of them (the same ones google_base uses). Apple takes no part: it has no depth map to be placed by. Meant to be placed on the Google base rather than by links.

- 4. Placement (postprocess/place.py, CPU only) -- runs right after reconstruction; the first half of post-processing, step 6 the second
Input: the run's scene -- its centre and one piece per DA3 frame.

A piece arrives internally consistent but individually placed: its own DA3 frame has an arbitrary origin, rotation and scale. A piece is the nodes sharing a DA3 frame -- a connected component of the scene's edges, derived on demand rather than stored. `postprocess.pipeline.process` is all of post-processing as one call: placement, then the fill (step 6; `fill=False` / `--no-fill` stops after placement).

    - 4.1 Scale, once per scene: the median over pieces whose cameras span at least 8 m of their DA3 camera spacing against their GPS spacing, falling back to `config.DA3_UNITS_TO_METRES`. Never per piece: that turns GPS noise into pieces of different sizes.
    - 4.2 One rigid fit per piece. Every camera's DA3 centre onto its GPS point, 2.45 m above its pano's elevation, and its DA3 orientation onto its pano's own heading, pitch and roll -- both at once, least squares, weighed by how far each is trusted (0.5 m, 2 deg). The photo keeps the camera's tilt (a car on a slope, a backpack leaning 8 deg on a flat street), so pitch and roll are what stand a piece upright; roll's sign is the opposite of pitch's (measured on NTU). A lone pano lands exactly on its GPS point and orientation. On NTU and Stockholm every camera came out within 0.2 m of its elevation and 0.8 deg of its pano's orientation.
    - 4.3 Floating bits (blobs.py). All placed points go into 10 cm cells, cells within 25 cm are joined, and a joined group under 300 cells -- a clump touching nothing else, like the remains of a half-masked lamp -- is removed. Across the whole scene, so a clump touching another node's surface stays. On NTU: 0.7% of the points; the few bigger loose pieces (ground patches, a far building) stay.

This replaced a road-by-road alignment (road lines, sliding pieces onto them, matching kerbs across the road, a fitted elevation surface, tilt from DA3's own ground). It levelled pieces from their ground and never used pitch or roll; the one fit does better with none of it.

Output: a `transform` on every placed node -- its own stored .ply straight to world metres. Saved into the scene rather than baked into the clouds, so rendering any subset (`postprocess.render_pieces`) is a matrix multiply with nothing re-solved.

- 5. Google base (google_base/, CPU) -- built per scene, used by step 6
Input: a scene. Output: a clean shape of the street from Google's own depth maps -- shape only, colour comes later from painting -- meant as the reference DA3 is fitted onto and whose surfaces fill DA3's gaps. `python -m streetview_to_3d.google_base SCENE_DIR OUT_DIR` writes it as a scene folder (one piece per Google pano, one for the ground).

    - 5.1 Panos (fetch.py). The scene's Google nodes plus official neighbours within 15 m and 5 years of the scene's date. Each is placed by its own GPS, elevation, heading, pitch and roll (`postprocess.place.photo_from_world`, the same as placement) and nothing else: nudging panos to agree with each other was tried and made walls worse. A depth map keeps the camera's tilt like the photo: with heading alone, Stockholm's walls leaned 2.3 deg typically and up to 9; with pitch and roll they stand exactly upright.
    - 5.2 Own planes (build.py). A Google depth map IS a list of planes plus a plane number per pixel (`services.streetview_fetch.fetch_depth_planes` keeps the numbers streetlevel throws away), so every ray lands exactly on its own plane. Only the dense part is kept: where one depth pixel covers under 45 cm of surface.
    - 5.3 Walls merged. Planes of different panos that are the same surface become one plane between them. Floors take no part -- moving some of a floor's planes and not their neighbours tears it.
    - 5.4 One ground. Every pano's ground (`postprocess.ground`, the one ground detector: faces up, lowest in its spot, connected to where the cameras stand -- slopes included) goes into one height map, each spot taken from the nearest camera's pano, and the ground is rebuilt on it as an even 5 cm grid. Google draws a level floor 2.5 m under every camera, so on a slope neighbouring panos' floors stack; the one map removes that.

- 6. Fill (fill/, CPU) -- the end of post-processing, right after placement, on the scene in place
Input: a placed scene. DA3's shape is never moved; its ground is replaced and its gaps filled, and every added point is written into the node whose pano coloured it (points belong to nodes), so the viewer needs nothing new. `python -m streetview_to_3d.fill SCENE_DIR [OUT_DIR]`.

    - 6.1 One ground (one_ground.py). Every cloud's ground (`postprocess.ground`) goes into one height map (`postprocess.ground.GroundMap`, the same one step 5.4 uses): each square takes the nearest camera's ground; where DA3 has none, Google's ground fills in, shifted onto DA3's height. The area is every square with ground, gaps closed, plus each camera's blind disc -- DA3's views stop ~29 deg down, so ~4.5 m under each camera is empty -- out to where its views stop but never past the nearest wall. It is laid out as an even 5 cm grid, and DA3's own points lying on it are removed.
    - 6.2 Google's walls (google.py). Google panos whose depth disagrees with DA3's are not used. From the rest, Google's walls fill where DA3 has nothing along the line of sight, reaching slightly into DA3's edge. Only near-vertical, trimmed, not-tiny surfaces. Where DA3 has the same wall, DA3 wins: the fill slides onto DA3's plane (no step), and is kept only along the stretches where DA3 really has that wall (a 0.6 m run with too few DA3 points is a doorway or the wall's end) -- sideways never past it, upward freely.
    - 6.3 Colour (paint.py). DA3 keeps its own colours. A point is kept only if some camera has it in view (nothing of DA3's in front, whole sphere); floor behind a fence no camera saw is left out. Its colour comes patch by patch from the nearest camera that may colour it (not masked, not in the blur many Google panos have below the horizon over the capture rig, not more than 70 deg down); the spot right under a camera, in view but not colourable, takes the painted ground around it.

Coordinates are Y-DOWN throughout (+X east, +Y down, +Z north, metres from the scene's centre), which is DA3's own convention; the viewer flips it only for display. Anything from outside must be converted: Google's `elevation` is metres above sea level (Y-up), so it is negated on the way in (see `road_align/ground_elevation.py`).
