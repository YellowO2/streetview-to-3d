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


    - 3.2 Solo mode (solo.py) -- the "Link panoramas" toggle off. No walk: every Google pano is reconstructed on its own, one piece each. The panos are each route dot's Google candidates (best-ranked date within 5 years of the others) plus Google's official neighbours of them. Apple takes no part. Each placed by its own GPS, elevation and orientation rather than by links.

- 4. Placement (postprocess/place.py, CPU only) -- runs right after reconstruction; the first half of post-processing, step 6 the second
Input: the run's scene -- its centre and one piece per DA3 frame.

A piece arrives internally consistent but individually placed: its own DA3 frame has an arbitrary origin, rotation and scale. A piece is the nodes sharing a DA3 frame -- a connected component of the scene's edges, derived on demand rather than stored. `postprocess.pipeline.process` is all of post-processing as one call: placement, then the fill (step 6; `fill=False` / `--no-fill` stops after placement).

    - 4.1 Scale, per piece: a piece whose cameras span at least 8 m fits its DA3 camera spacing against its GPS spacing; every other piece (lone panos included) takes the median of those, falling back to `config.DA3_UNITS_TO_METRES`. One scale for the whole scene left pieces mismatched in size (each is its own DA3 run); Google's depth maps as the measure looked worse.
    - 4.2 One rigid fit per piece. Every camera's DA3 centre onto its GPS point, 2.45 m above its pano's elevation, and its DA3 orientation onto its pano's own heading, pitch and roll -- both at once, least squares, weighed by how far each is trusted (0.5 m, 2 deg). The photo keeps the camera's tilt (a car on a slope, a backpack leaning 8 deg on a flat street), so pitch and roll are what stand a piece upright; roll's sign is the opposite of pitch's (measured on NTU). A lone pano lands exactly on its GPS point and orientation. On NTU and Stockholm every camera came out within 0.2 m of its elevation and 0.8 deg of its pano's orientation.
    - 4.3 Floating bits (blobs.py). All placed points go into 10 cm cells, cells within 25 cm are joined, and a joined group under 300 cells -- a clump touching nothing else, like the remains of a half-masked lamp -- is removed. Across the whole scene, so a clump touching another node's surface stays. On NTU: 0.7% of the points; the few bigger loose pieces (ground patches, a far building) stay.

This replaced a road-by-road alignment (road lines, sliding pieces onto them, matching kerbs across the road, a fitted elevation surface, tilt from DA3's own ground). It levelled pieces from their ground and never used pitch or roll; the one fit does better with none of it.

Output: a `transform` on every placed node -- its own stored .ply straight to world metres. Saved into the scene rather than baked into the clouds, so rendering any subset (`postprocess.render_pieces`) is a matrix multiply with nothing re-solved.

- 5. Fill (fill/, CPU) -- right after placement, on the scene in place
Input: a placed scene. DA3's shape is never moved; its ground is replaced and its holes filled, and every added point is written into the node whose pano coloured it (points belong to nodes), so the viewer needs nothing new. `python -m streetview_to_3d.fill SCENE_DIR [OUT_DIR]`.

    - 5.1 One ground (one_ground.py). Every cloud's ground (`postprocess.ground`) goes into one height map (`postprocess.ground.GroundMap`): each square takes the nearest camera's ground. The area is every square with ground, gaps closed, holes filled, plus each camera's blind disc -- DA3's views stop ~29 deg down, so ~4.5 m under each camera is empty -- out to where its views stop but never past the nearest wall. It is laid out as an even 5 cm grid, and DA3's own points lying on it are removed.
    - 5.2 Colour (paint.py). DA3 keeps its own colours. A point is kept only if some camera has it in view (nothing of DA3's in front, whole sphere); floor behind a fence no camera saw is left out. Its colour comes patch by patch from the nearest camera that may colour it (not masked, not in the blur many Google panos have below the horizon over the capture rig, not more than 70 deg down); the spot right under a camera, in view but not colourable, takes the painted ground around it.

    Google's depth maps once filled in too -- their ground where DA3 had none, their walls above DA3's reach. The terrain's land and OSM buildings (step 6) do that now with no download, so they are gone.

- 6. Surroundings (postprocess/terrain.py, CPU) -- the end of post-processing
Input: a placed, filled scene. Output: terrain.ply (the land and roads) and buildings.ply beside scene.json, already in the world frame; the viewer loads both with the scene. Built once, so a download is never needed to view it.

    - 6.1 Land: AWS Terrain Tiles (no key, ~30 m SRTM), 1 km out or 2 km where hills rise beyond; shifted onto Google's datum, then bent near the panos onto each pano's own elevation (SRTM was up to 4 m off on NTU, where Google's matched DA3's own slope). It runs on under the scene just beneath its ground, and meets and takes the scene's colour at its edge (seams.py). Coloured by EOX's Sentinel-2 cloudless (credit it; non-commercial).
    - 6.2 OpenStreetMap (osm.py): buildings within 1 km and roads within 700 m in one Overpass request, mirrors tried in turn, cached as osm.json. Roads (roads.py) laid on the land by width and coloured by surface.
    - 6.3 Buildings (buildings.py): each outline raised by its height tag, else floors, else a guess. One DA3 built a wall of is slid onto it (DA3's own plane for that wall, points facing its way) and cut back, as a solid block, where it still stands in front of it; never onto a road. What DA3 already has of it is left to DA3, judged on the wall itself, the rest pulled onto DA3's plane at the seam. Inner walls of sparser points keep it from being seen through. Colour: what the panos see of it, else its colour tag, else the place's own building colours, softened.
    - 6.4 Spacing: every map point is 5 cm from the next at a camera, 1.8% of the distance to the nearest camera more further out; the viewer draws them to match. Near the cameras everything is painted from the panos as the fill's ground is.

Coordinates are Y-DOWN throughout (+X east, +Y down, +Z north, metres from the scene's centre), which is DA3's own convention; the viewer flips it only for display. Anything from outside must be converted: Google's `elevation` is metres above sea level (Y-up), so it is negated on the way in (see `road_align/ground_elevation.py`).
