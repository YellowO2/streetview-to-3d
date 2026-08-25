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

    - 2.1 Build the fetch_graph  (fetch_nodes.py: `corridor_points`, `fetch_corridor_nodes`)
    The fetch_graph's dots ARE the selection_graph's own real nodes -- `corridor_points` builds the dot/adjacency structure directly from the corridor's real edges, no synthetic in-between sampling. For each dot, `fetch_corridor_nodes` fetches the nearby Apple and Google panoramic metadata (no images yet), within `POINT_MAX_DIST_M` = 5m of that dot's own real coordinates.
    Output: a fetch_graph where each dot has a bucket of candidate panoramic metadata (no images).

    - 2.2 Build the top N date_graphs  (build_graph.py: `build_corridor_graphs`; date_ranking.py)
    The fetch_graph's candidates span multiple real-world capture dates. We split it into isolated graph per date -- e.g. if the fetch_graph's contains dates A, B and C, we get 3 separate `date_graphs`.
        - 2.2.1 Select top N date_graphs (currently N = `DATE_TOP_N` = 3)
        `rank_dates` scores every candidate date by coverage span (earliest-to-latest dot it reaches), then total dot count, then recency as a tiebreaker. `date_connects` then filters to dates that can structurally reach from the start zone toward a goal (walking dot-to-dot through the fetch_graph's adjacency, ONLY through non-empty directly-adjacent dots -- no flood past an empty one) -- a date only counts toward the top N if it passes this reachability check, not just the ranking.
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
        - 3.1.4 If 3.1.3 fails: no flood/skip fallback. A dot is a real selection-graph node, so a failed or empty direct structural neighbor is a genuine dead end for that date -- the neighbor is simply left unconfirmed.
        - 3.1.5 On any merge success, the neighbor is queued so its own onward neighbors get tried next.
        - 3.1.6 Repeat 3.1.1-3.1.5 until every dot is covered or visited, then move to the next date_graph.
        - 3.1.8 `set_cover`: across all dates' pieces, greedily pick the fewest that cover the most of the corridor. `protected_positions` (a chunked large-area reconstruction's own known cross-chunk boundary node COORDINATES) are force-kept afterward even if `set_cover` would otherwise drop their piece as geographically redundant. Matched by exact dot INDEX, not node key or distance -- every date graph walks the same `points`/`adjacency` object, so a dot index is a precise, date-independent identity for "this real location"; a node KEY is date-specific (the same real spot gets a different pano id per historical date), so exact-key matching could never rescue a location whose winning date differs from wherever the coordinate snapshot came from. A protected position genuinely absent from every one of the top-N walked dates (see 2.2.1 -- it may still have OTHER real dates, just none among the few actually downloaded) simply stays unrescued.
        - 3.1.9 Self-bridge: still inside the SAME GPU call as 3.1.1-3.1.8 (no extra `@spaces.GPU` call), `bridge_pieces` runs once more, blind (no chunk-pair restriction) over 3.1's own chosen segments -- catches connections the walk's narrower structural-neighbor-only search missed, before this chunk's result is ever saved or hands off to cross-chunk bridging.

    Output: **segments** -- `[(points, colors, path_edges, date, reached_all, node_positions, frame_poses), ...]`, one per chosen piece (already self-bridged per 3.1.9). Usually more than one if no single date's coverage alone spans the whole corridor.
    
    - 3.2 Join (join_segments.py: `join_segments`, `bridge_pieces`) -- the "Join" button
    Segments from 3.1 are each in their own arbitrary DA3-local frame (nothing ties them together yet). Join tries to stitch them: for every candidate pair of segments (or only declared-adjacent pairs, if reconstructing many chunks -- see `known_adjacent_chunk_pairs`), pick the closest real-lat/lon node pair as bridge candidates and run a real DA3 test between them; success chains a rigid-align onto the existing piece and merges the two segments into one. No GPS placement is used -- only real DA3-confirmed connections merge anything.
    Output: **pieces** -- `[(points, colors, metadata), ...]`, one per still-separate result. metadata is `{node_key: {lat, lon, date}}`. More than one piece means bridging genuinely couldn't connect everything (not an error, unless a *declared* adjacency had zero candidates in range at all -- see `NoBridgeCandidatesError`).

- 4. Large-area chunked reconstruction (CLI, street_builder/tab.py's "Scripted staged testing" section + tests/staged_corridor_test.py)
A whole campus is too large for one GPU call, so it's split into chunks and run one at a time, incrementally bridged onto whatever's already merged so far -- not generated all at once then joined once at the end.

    - 4.0 Whole-area date cover (build_graph/global_dates.py, offline/one-time: `tests/fetch_ntu_metadata.py` then `tests/inspect_global_date_cover.py`) -- replaces 2.2's per-chunk top-N date ranking for a large-area run. `build_date_cover` assigns EVERY dot in the whole area to exactly one date (seeded from the single best-ranked date, gaps greedily filled by whichever other date's own contiguous run overlaps the gap most, adopting that run's full span -- not just the missing dots -- to minimize how many times the assignment switches date). This exists because independent per-chunk date ranking was a real, recurring cause of cross-chunk bridging failure: two adjacent chunks could each rank a different date best locally even when both had real data on a shared date. `split_cover_into_chunks` then cuts chunks (~20 dots each) directly from this cover, growing each chunk only through same-date neighbors -- so a chunk is single-date BY CONSTRUCTION and a cross-date seam only ever happens AT a chunk boundary, exactly where 4.2's bridging already handles it.
    - 4.1 `cli_run_chunk` -- Prepare (`street_main.prepare_pathfind_from_cover_chunk`, sourcing straight from 4.0's cover by dot index -- no per-dot date lookup needed, the chunk's date is already fixed) + Run for ONE chunk (its own `@spaces.GPU` call). Saves this chunk's own raw, un-bridged segments to a dataset repo (`CLI_RAW_PREFIX/<chunk_id>`) immediately, independent of whatever bridging does with them next -- so a bug in bridging never requires re-running this (expensive) step, only re-running the bridge against the same saved data. (An area outside 4.0's coverage falls back to `street_main.prepare_pathfind` + `map_selection.candidates.split_into_chunks`'s raw-graph chunking instead.)
    - 4.2 `cli_bridge_chunk` -- its own separate `@spaces.GPU` call (never combined with 4.1: two `@spaces.GPU` calls in one request can let the ZeroGPU proxy token go stale between them). Downloads the current checkpoint from the dataset, downloads this chunk's own saved raw segments, calls `bridge_incremental_gpu` (only tests the new chunk against declared-adjacent existing chunks, never re-verifies pairs already merged in an earlier call), then re-uploads the merged result as the new checkpoint.
    - 4.3 The checkpoint (`.ply` + metadata `.json` per still-separate piece, under `CLI_CHECKPOINT_PREFIX`) IS the live, resumable state -- the same files are both directly viewable and exactly what a later `cli_bridge_chunk` call reads back in (`output_to_piece`, join_segments.py) to keep bridging further. No separate internal format, no separate "finalize" step.

