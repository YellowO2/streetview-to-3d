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
    The fetch_graph's dots ARE the selection_graph's own real nodes -- `corridor_points` builds the dot/adjacency structure directly from the corridor's real edges, no synthetic in-between sampling. For each dot, `fetch_corridor_nodes` fetches the nearby Apple and Google panoramic metadata (no images yet), within `POINT_MAX_DIST_M` = 5m of that dot's own real coordinates.
    Output: a fetch_graph where each dot has a bucket of candidate panoramic metadata (no images), plus that dot's ground elevation, which the same lookup already returned -- see 4.4.

    - 2.2 Build the top N date_graphs  (build_street_graph/build_graph.py: `build_corridor_graphs`; date_ranking.py)
    The fetch_graph's candidates span multiple real-world capture dates. We split it into isolated graph per date -- e.g. if the fetch_graph's contains dates A, B and C, we get 3 separate `date_graphs`.
        - 2.2.1 Select top N date_graphs (currently N = `DATE_TOP_N` = 3)
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

        - 3.1.1 Pick a seed (`pick_seed`) from THIS date_graph's closest dot to `start_node` in step 1. Only called when the BFS queue is empty but the corridor still isn't fully covered or visited (normal in-progress BFS traversal, popping the queue, is a separate branch that never calls this). Every later seed: the not-yet-visited dot closest to a still uncovered region. curr_node = start_node.
        - 3.1.2 Rate a node. When curr_node not in visited, we run `rate_node`, which does DA3 scoring on every one of node B candidate panos, discard failed ones, but always keep at least one single best-scoring candidate (even if failed) + its solo DA3 point cloud. Then mark the dot `visited`.
        - 3.1.3 Connecting to a neighbour. Set curr_node to direct structural neighbor (typical bfs with queue), rate_node it too if not yet visited, then run a real DA3 `test_edge` between the two dots' best candidates. On success, `test_edge` itself returns a jointly-computed point cloud for the pair (higher quality then solo pieces via 'rate_node'). If either dot was already part of a larger piece (from earlier successful connections), the pairwise pointcloud result of only the NEW NODE (as one node might already be in) is joined onto that existing piece instead.
        - 3.1.4 If 3.1.3 fails: no flood/skip fallback. A dot is a real selection-graph node, so a failed or empty direct structural neighbor is a genuine dead end for that date -- the neighbor is simply left unconfirmed.
        - 3.1.5 On any merge success, the neighbor is queued so its own onward neighbors get tried next.
        - 3.1.6 Repeat 3.1.1-3.1.5 until every dot is covered or visited, then move to the next date_graph.
        - 3.1.8 `set_cover`: across all dates' pieces, greedily pick the fewest that cover the most of the corridor. `protected_positions` (a chunked large-area reconstruction's own known cross-chunk boundary node COORDINATES) are force-kept afterward even if `set_cover` would otherwise drop their piece as geographically redundant. Matched by exact dot INDEX, not node key or distance -- every date graph walks the same `points`/`adjacency` object, so a dot index is a precise, date-independent identity for "this real location"; a node KEY is date-specific (the same real spot gets a different pano id per historical date), so exact-key matching could never rescue a location whose winning date differs from wherever the coordinate snapshot came from. A protected position genuinely absent from every one of the top-N walked dates (see 2.2.1 -- it may still have OTHER real dates, just none among the few actually downloaded) simply stays unrescued.
        - 3.1.9 Self-bridge: still inside the SAME GPU call as 3.1.1-3.1.8 (no extra `@spaces.GPU` call), `bridge_pieces` runs once more, blind (no chunk-pair restriction) over 3.1's own chosen segments -- catches connections the walk's narrower structural-neighbor-only search missed, before this chunk's result is ever saved or hands off to cross-chunk bridging.

    Output: **segments** -- `[(clouds, path_edges, date, reached_all, node_positions, frame_poses), ...]`, one per chosen piece. `clouds` is `{node key: (points, colors)}`: DA3 reconstructs at most two panoramas at a time and a node's points enter exactly once, so every point belongs to a known node (already self-bridged per 3.1.9). Usually more than one if no single date's coverage alone spans the whole corridor.
    
    - 3.2 Join (join_segments.py: `join_segments`, `bridge_pieces`) -- the "Join" button
    Segments from 3.1 are each in their own arbitrary DA3-local frame (nothing ties them together yet). Join tries to stitch them: for every candidate pair of segments (or only declared-adjacent pairs, if reconstructing many chunks -- see `known_adjacent_chunk_pairs`), pick the closest real-lat/lon node pair as bridge candidates and run a real DA3 test between them; success chains a rigid-align onto the existing piece and merges the two segments into one. No GPS placement is used -- only real DA3-confirmed connections merge anything.
    Output: **pieces** -- `[(points, colors, metadata), ...]`, one per still-separate result. metadata is `{node_key: {lat, lon, date, position, rotation, n_views_kept, n_views_total, links}}` -- `links` being this node's DA3-confirmed neighbours and the keep counts from each joint test. More than one piece means bridging genuinely couldn't connect everything (not an error, unless a *declared* adjacency had zero candidates in range at all -- see `NoBridgeCandidatesError`).

    Each piece is recorded in the run's `scene.json` as it is saved (see `scene.py`), which is what step 4 reads.

- 4. Placement (postprocess/, CPU only) -- the "Place into one scene" button
Input: the run's scene -- its centre, its street graph, and one Piece per cloud.

A piece arrives internally consistent but individually placed: its own DA3 frame has an arbitrary origin, rotation and scale, and nothing ties it to the ground or to its neighbours. `postprocess.pipeline.process` is the whole stage as one call.

    - 4.1 Fit to GPS (gps_fit/load_pieces.py). Each node carries both its DA3 position and its real lat/lon, so a 2D similarity fit (rotation and translation) puts the piece on the map. Height is NOT fitted here -- the fit is top-down only. Scale is not fitted at all: DA3 is internally consistent, so one measured constant converts its units to metres (`config.DA3_UNITS_TO_METRES`, see the README).
    - 4.2 Split where the fit fails (gps_fit/split_by_fit.py). A piece is placed by ONE rigid transform, so a piece that bent during reconstruction cannot be placed at all. Nodes further than `FIT_THRESHOLD_M` from their own GPS are cut off and the fragments re-fitted, turning a bend inside a piece into a gap between two pieces. Nothing is copied: points belong to nodes, so this only rewrites which nodes are grouped together.
    - 4.3 Road alignment (road_align/node_center_to_road_line.py). Road centre lines are built from the scene's own street graph (corridors.py chains dots into roads). Each piece slides sideways onto its road's line. A piece with more than two nodes may only translate -- GPS already measured its heading; one with two or fewer may also rotate, since nothing ever constrained it.
    - 4.4 Height (road_align/ground_elevation.py). Nothing in a reconstruction says how high a piece sits, so the ground itself supplies the datum. Every pano lookup returns an elevation, so step 2.1 already recorded one per dot and the scene's graph carries them -- no second round of network calls, and coverage over the whole area rather than only the dots that reconstructed. They are metres above sea level (Y-UP) and are negated on the way in (see the README's coordinate convention). A smooth surface is fitted through them and each piece gets one height offset and one tilt onto it.

Output: `piece_transforms.json` -- one 4x4 per piece, stored rather than baked in, so any subset can be rendered without solving again.
