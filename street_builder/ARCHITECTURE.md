This is a pipeline that allows users to select a region on a map, then obtain a 3d point cloud representation of it.
It can be seen as google map -> 3d point cloud.

The architecture is as follows:

- 1. map_selection  (street_builder/map_selection/)
Enables the user to select the region on the map, either by clicking nodes one by one or via a radius. BFS is used to guarenteed a connected graph.
The start_node is the closest real google node to the inputed lon-lat of user.
Output: a `selection_graph` -- the raw Google-only nodes + edges the user selected. In code this lives as `state["selected"]` (node keys) + `state["selected_edges"]` (edges).`selection_graph` is the term we use to talk about that pair together.

- 2. build_graph  (street_builder/build_graph/)
Input: `selection_graph`

With the `selection_graph` from map_selection, we do a series of processing to build our own graph.

    - 2.1 Build the fetch_graph  (fetch_nodes.py: `interpolate_points`, `fetch_corridor_nodes`)
    We make the selection_graph finer: `interpolate_points` resamples each edge into points spaced `POINT_SPACING_M` = 5m apart (vs. the selection_graph's native ~10m). This finer graph is called the **fetch_graph** because for each of its points, `fetch_corridor_nodes` fetches the nearby Apple and Google panoramic metadata (no images yet).
    Output: a fetch_graph where each dot has a bucket of candidate panoramic metadata (no images).

    - 2.2 Build the top N date_graphs  (build_graph.py: `build_corridor_graphs`; date_ranking.py)
    The fetch_graph's candidates span multiple real-world capture dates. We split it into isolated graph per date -- e.g. if the fetch_graph's contains dates A, B and C, we get 3 separate `date_graphs`.
        - 2.2.1 Select top N date_graphs (currently N = `DATE_TOP_N` = 3)
        `rank_dates` scores every candidate date by coverage span (earliest-to-latest dot it reaches), then total dot count, then recency as a tiebreaker. `date_connects` then filters to dates that can structurally reach from the start zone toward a goal (walking dot-to-dot through the fetch_graph's adjacency, hopping past empty dots up to `EDGE_MAX_DIST_M` = 18m) -- a date only counts toward the top N if it passes this reachability check, not just the ranking.
        - 2.2.2 Fetch panoramic images for the top N date_graphs
        We only download images for the top N date_graphs, to avoid fetching images for dates we'll never use.
    Output: N date_graphs, each with metadata + downloaded pano image per dot.

Output: the output of 2.2 (the top N date_graphs) feeds into step 3.

- 3. 3d reconstruction  (street_builder/reconstruction/)
Input: the N date_graphs from step 2.
Turns downloaded panos into an actual 3d point cloud, via real pairwise DA3 tests between panos.

    - 3.1 Corridor pathfind (walk_graph.py: `run_pathfind_reconstruction`)
    Per date_graph, independently walk and grow the dots in a bfs fashion. Two concepts drive this: `visited` dict (has this dot been rated/visited yet) and `covered` (is this dot's location already within `point_cover_tolerance_m`=15m of some visited dot. Currently computed via `covered_points()`. A dot can be covered without itself being visited).

        - 3.1.1 Pick a seed (`pick_seed`) from THIS date_graph's closest dot to `start_node` in step 1. Only called when the BFS queue is empty but the corridor still isn't fully covered or visited (normal in-progress BFS traversal, popping the queue, is a separate branch that never calls this). Every later seed: the not-yet-visited dot closest to a still uncovered region. curr_node = start_node.
        - 3.1.2 Rate a node. When curr_node not in visited, we run `rate_node`, which does DA3 scoring on every one of node B candidate panos, discard failed ones, but always keep at least one single best-scoring candidate (even if failed) + its solo DA3 point cloud. Then mark the dot `visited`.
        - 3.1.3 Connecting to a neighbour. Set curr_node to direct structural neighbor (typical bfs with queue), rate_node it too if not yet visited, then run a real DA3 `test_edge` between the two dots' best candidates. On success, `test_edge` itself returns a jointly-computed point cloud for the pair (higher quality then solo pieces via 'rate_node'). If either dot was already part of a larger piece (from earlier successful connections), the pairwise pointcloud result of only the NEW NODE (as one node might already be in) is joined onto that existing piece instead.
        - 3.1.4 If 3.1.3 fails entirely: flood past the neighbor, up to `EDGE_MAX_DIST_M` (18m) from the original dot, trying each reachable candidate dot closest-first. First success confirms into the same piece.
        - 3.1.5 On any merge success, the neighbor is queued so its own onward neighbors get tried next.
        - 3.1.6 Repeat 3.1.1-3.1.5 until every dot is covered or visited, then move to the next date_graph.
        - 3.1.8 `set_cover`: across all dates' pieces, greedily pick the fewest that cover the most of the corridor.

    Output: **segments** -- `[(points, colors, path_edges, date, reached_all, node_positions, frame_poses), ...]`, one per chosen piece. Usually more than one if no single date's coverage alone spans the whole corridor.
    
    - 3.2 Join (join_segments.py: `join_segments`, `bridge_pieces`) -- the "Join" button
    Segments from 3.1 are each in their own arbitrary DA3-local frame (nothing ties them together yet). Join tries to stitch them: for every candidate pair of segments (or only declared-adjacent pairs, if reconstructing many chunks -- see `known_adjacent_chunk_pairs`), pick the closest real-lat/lon node pair as bridge candidates and run a real DA3 test between them; success chains a rigid-align onto the existing piece and merges the two segments into one. No GPS placement is used -- only real DA3-confirmed connections merge anything.
    Output: **pieces** -- `[(points, colors, metadata), ...]`, one per still-separate result. metadata is `{node_key: {lat, lon, date}}`. More than one piece means bridging genuinely couldn't connect everything (not an error, unless a *declared* adjacency had zero candidates in range at all -- see `NoBridgeCandidatesError`).

