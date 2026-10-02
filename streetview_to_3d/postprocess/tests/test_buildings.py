import numpy as np

from streetview_to_3d.postprocess import buildings
from streetview_to_3d.postprocess.roofs import Roof


def _box(x0, z0, size=10.0):
    xy = np.array([[x0, z0], [x0 + size, z0], [x0 + size, z0 + size], [x0, z0 + size], [x0, z0]], float)
    return (xy, 9.0, False, buildings.Form(Roof(xy, "flat")))


def _settle(outlines, slope):
    """The land a grid 1 m apart rising slope per metre east, plus every
    outline's corners, settled."""
    gx, gz = np.meshgrid(np.arange(-20, 60, 1.0), np.arange(-20, 40, 1.0))
    grid = np.c_[gx.ravel(), gz.ravel()]
    walls, owner = buildings.corners(outlines, lambda xy: np.full(len(xy), 1.0))
    xy = np.concatenate([grid, walls])
    own = np.r_[np.full(len(grid), -1), owner]
    return xy, own, buildings.settle(outlines, xy, xy[:, 0] * slope, own)


def test_settled_building_is_never_buried_nor_floating():
    # a house on a 10 % slope: the map's ground rises 1 m across it
    house = _box(0, 0)
    xy, own, h = _settle([house], 0.1)
    form = house[3]
    assert form.foot_m == 0.0                                   # the land's lowest along it
    assert np.allclose(h[own == 0], form.foot_m)                # level with its foot all round
    assert form.skirt_m == 0.0
    far = np.linalg.norm(xy - [30, 5], axis=1) < 1               # 20 m off: the land its own
    assert np.allclose(h[far], xy[far, 0] * 0.1)
    assert (h <= xy[:, 0] * 0.1 + 1e-9).all()                   # only ever cut down


def test_lower_neighbours_cut_is_reached_by_the_higher_ones_walls():
    # two houses side by side up a steep slope: the lower one's cut reaches the higher one
    low, high = _box(0, 0), _box(10.5, 0)
    xy, own, h = _settle([low, high], 0.3)
    assert high[3].foot_m > low[3].foot_m and high[3].skirt_m > 1.0
    assert high[3].foot_m - high[3].skirt_m <= h[own == 1].min() + 1e-9    # its walls reach the land
    # its floors counted from its own foot: the far solid's facade says so
    v, _, _, facade = buildings.solid([high], lambda p: np.zeros(len(p)), np.ones((1, 3)) * 0.8)
    wall = facade[:, 1] > -1000
    assert np.isclose((-v[wall, 1]).min(), high[3].foot_m - high[3].skirt_m)
    assert np.isclose(facade[wall, 1].min(), -high[3].skirt_m)


def test_facades_follow_storeys_and_building_kind_and_export_layout(tmp_path):
    from streetview_to_3d.postprocess.ply_io import write_mesh, read_ply
    house = _box(0, 0)
    house[3].tags = {"building": "house", "building:levels": "3"}
    office = _box(20, 0)
    office[3].tags = {"building": "office", "building:levels": "2"}
    assert buildings.facade_layout(house[3], 9) == (2.8, 3.0)
    assert buildings.facade_layout(office[3], 9) == (2.4, 4.5)
    v, c, f, facade, gaps = buildings.solid([house, office], lambda xy: np.zeros(len(xy)),
                                           np.full((2, 3), .8), [2.0, 2.5])
    wall = facade[:, 1] > -1000
    assert set(facade[wall, 2].round(1)) == {np.float32(2.8), np.float32(2.4)}
    assert np.isfinite(v).all() and f.max() < len(v)
    path = tmp_path / "blocks.ply"
    write_mesh(path, v, c, f, facade, gaps)
    loaded, colours = read_ply(path)
    assert np.allclose(v, loaded) and np.allclose(c, colours, atol=1 / 255)
    assert b"property float facade_bay" in path.read_bytes()


def test_details_respect_height_and_skip_reconstructed_walls():
    house = _box(0, 0)
    xy, h, _, form = house
    ground = lambda xy: np.zeros(len(xy))
    detail = list(buildings.detail_quads(xy, h, form, ground, np.full(3, .8)))
    assert len(detail) == 10  # two trims per wall and door/canopy on one wall
    assert max(-quad[:, 1].min() for quad, _ in detail) <= h
    assert not list(buildings.detail_quads(xy, h, form, ground, np.full(3, .8), dict.fromkeys(range(4), True)))
    blocks = buildings.points([(*house, {})], lambda xy: np.full(len(xy), .4), ground,
                               np.full((1, 3), .8))
    buildings.windows(blocks, [house], ground)
    assert all(len(getattr(blocks, k)) == len(blocks.pts) for k in ("cols", "edge", "gap", "own"))
    assert np.isfinite(blocks.pts).all() and np.isfinite(blocks.cols).all()
    assert (blocks.cols >= 0).all() and (blocks.cols <= 1).all()


def test_physical_apartments_change_silhouette_and_have_no_painted_window_grid():
    from streetview_to_3d.postprocess.facade_geometry import profile_for
    house = _box(0, 0)
    house[3].tags = {"building": "apartments", "building:levels": "3"}
    ground = lambda xy: np.zeros(len(xy))
    colour = np.full((1, 3), .8)
    v, c, f, facade, gap = buildings.solid([house], ground, colour, [.3])
    # far off, solid: its outline's walls and roof only, its windows paint on its facade
    assert v[:, 0].min() >= 0 and v[:, 2].min() >= 0
    assert (facade[:, 1] > buildings.NO_FACADE).any()
    assert f.max() < len(v) and np.isfinite(v).all() and np.isfinite(c).all()
    assert profile_for({"building": "apartments", "start_date": "1890"}, 15, 200).name == "historic_urban"
    assert not profile_for({"building": "office"}, 15, 200).balcony
    assert profile_for({"building": "apartments"}, 15, 200).balcony
    blocks = buildings.points([(*house, {})], lambda xy: np.full(len(xy), .3), ground, colour)
    before = blocks.cols.copy()
    buildings.windows(blocks, [house], ground)
    assert np.array_equal(before, blocks.cols)  # never print a second grid beneath physical windows
    assert blocks.pts[:, 0].min() < -.9


def test_shared_walls_and_airborne_parts_do_not_get_ground_entrances():
    from streetview_to_3d.postprocess.facade_geometry import facade_quads
    house = _box(0, 0)
    xy, h, _, form = house
    form.tags = {"building": "apartments", "building:levels": "3"}
    form.shared_edges = frozenset({0, 1, 2, 3})
    # Roof equipment is still allowed; all facade ornaments are blocked.
    faces = list(facade_quads(xy, h, form, 0, np.full(3, .8)))
    assert faces and all((-q[:, 1]).min() >= h for q, _ in faces)
    form.geometry_cache.clear()
    form.part = True
    assert not list(facade_quads(xy, h, form, 0, np.full(3, .8)))


def test_parts_inherit_landmark_type_and_adjacent_walls_are_detected():
    from streetview_to_3d.postprocess.facade_geometry import enabled
    def element(i, x0, size, tags):
        xy = _box(x0, 0, size)[0]
        return {"type": "way", "id": i, "tags": tags,
                "geometry": [{"lon": x, "lat": y} for x, y in xy]}
    to_xy = lambda g: np.array([[p["lon"], p["lat"]] for p in g])
    parent = element(1, 0, 20, {"building": "church", "start_date": "1850"})
    part = element(2, 1, 10, {"building:part": "yes", "height": "9"})
    result = buildings.outlines([parent, part], to_xy)
    assert len(result) == 1 and result[0][3].tags["building"] == "church"
    assert not enabled(result[0][3], 9, .4)
    left = element(3, 0, 10, {"building": "apartments"})
    right = element(4, 10, 10, {"building": "apartments"})
    result = buildings.outlines([left, right], to_xy)
    assert 1 in result[0][3].shared_edges and 3 in result[1][3].shared_edges


def test_only_buildings_da3_reaches_take_the_panos_colour():
    near, away = _box(0, 0), _box(100, 0)
    ground = lambda xy: np.zeros(len(xy))
    blocks = buildings.points([(*near, {}), (*away, {})], lambda xy: np.full(len(xy), 1.0), ground,
                              np.full((2, 3), .8))
    from scipy.spatial import cKDTree
    da3 = np.array([[5.0, -4.0, -1.0]])  # a metre off the first's wall, 4 m up
    assert list(buildings.reached(blocks, cKDTree(da3), 2)) == [True, False]
    assert not buildings.reached(blocks, None, 2).any()


def test_points_know_their_facing_stroke_and_kind_their_windows_glass(tmp_path):
    from streetview_to_3d.postprocess.ply_io import write_ply
    house = _box(0, 0)
    ground = lambda xy: np.zeros(len(xy))
    blocks = buildings.points([(*house, {})], lambda xy: np.full(len(xy), .4), ground,
                              np.full((1, 3), .8))
    assert np.allclose(np.linalg.norm(blocks.normal, axis=1), 1)
    assert np.allclose(np.linalg.norm(blocks.along, axis=1), 1)
    assert np.allclose((blocks.normal * blocks.along).sum(1), 0, atol=1e-6)       # on its face
    # a roof (its normals east, north, up): flat, facing up, stroked east-west; sloping east, down it
    normal, along = buildings._roof_ways(np.array([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]]))
    assert np.allclose(normal, [[0, -1, 0], [0.6, -0.8, 0]])       # Blocks' y is down
    assert np.allclose(along[0], [1, 0, 0]) and along[1, 0] > 0 and along[1, 1] > 0
    wall = blocks.edge >= 0
    assert np.allclose(blocks.normal[wall, 1], 0)                 # a wall level
    edge = blocks.kind == buildings.EDGE
    assert edge.any() and (blocks.edge[edge] >= 0).all()          # corners and eaves, on walls
    assert (np.abs(blocks.along[edge, 1]) > 0.99).any()           # a corner's stroke goes up it
    house[3].physical_facade = False                                # no balconies of its own: windows painted
    buildings.windows(blocks, [house], ground)
    glass = np.all(np.isclose(blocks.cols, buildings.glass([.8, .8, .8])), axis=1)
    assert glass.sum() > 20 and (blocks.kind[glass] == buildings.SURFACE).all()   # the wall's points, glass
    assert np.allclose(blocks.cols[~glass & (blocks.edge >= 0)], .8)              # its walls as they are: unlit
    path = tmp_path / "buildings.ply"
    write_ply(path, blocks.pts, blocks.cols, blocks.gap, blocks.normal, blocks.along, blocks.kind)
    head = path.read_bytes()[:400]
    assert b"property float nx" in head and b"property float ax" in head and b"property uchar kind" in head


def test_points_turn_into_da3s_as_they_come_up_to_them():
    from scipy.spatial import cKDTree
    da3, da3_cols = np.array([[0.0, 0.0, 0.0]]), np.array([[1.0, 0.0, 0.0]])
    pts = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    cols, near, _ = buildings.toward(pts, np.full((3, 3), 0.5), np.full(3, .1), cKDTree(da3), da3_cols)
    assert near[0] == 1 and 0 < near[1] < 1 and near[2] == 0       # on one, half a metre, beyond BLEND_M
    assert np.allclose(cols[0], [1, 0, 0]) and np.allclose(cols[2], 0.5)
    assert np.allclose(buildings.toward(pts, np.full((3, 3), .5), np.full(3, .1), None, da3_cols)[1], 0)
    # a wall 10 cm apart meeting DA3's, 40 cm apart: on it, about as few as DA3's; a metre off, all
    g = np.arange(0, 6, .4)
    wall = np.c_[np.repeat(g, len(g)), np.tile(g, len(g)), np.zeros(len(g) ** 2)]
    f = np.arange(0, 6, .1)
    mine = np.c_[np.repeat(f, len(f)), np.tile(f, len(f)), np.zeros(len(f) ** 2)]
    _, near, keep = buildings.toward(mine, np.full((len(mine), 3), .5), np.full(len(mine), .1), cKDTree(wall),
                                     np.full((len(wall), 3), .5))
    on = near > 0.99
    assert 0.03 < keep[on].mean() < 0.12                            # (10/40)^2: about one in 16
    off = np.c_[mine[:, :2], np.full(len(mine), 1.5)]
    assert buildings.toward(off, np.full((len(off), 3), .5), np.full(len(off), .1), cKDTree(wall),
                            np.full((len(wall), 3), .5))[2].all()
