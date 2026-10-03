"""Street furniture (lamps, signals, benches, bins) as boxes for road points and meshes; lamps
also inferred along lit sidewalks."""
import numpy as np

METAL = (.24, .28, .29)
WOOD = (.48, .30, .17)


def _box(origin, direction, bounds, colour):
    """Closed six-face box in east/north/up local coordinates."""
    x0, x1, y0, y1, z0, z1 = bounds
    side = np.array([-direction[1], direction[0]])
    corners = []
    for x, y, z in ((x0,y0,z0),(x1,y0,z0),(x1,y1,z0),(x0,y1,z0),
                    (x0,y0,z1),(x1,y0,z1),(x1,y1,z1),(x0,y1,z1)):
        en = origin[:2] + x * direction + y * side
        corners.append([en[0], -(origin[2] + z), en[1]])
    corners = np.array(corners)
    for ids, shade in (((0,3,2,1),.65), ((4,5,6,7),1.1), ((0,1,5,4),.9),
                       ((1,2,6,5),.78), ((2,3,7,6),.85), ((3,0,4,7),.8)):
        yield corners[list(ids)], np.clip(np.array(colour) * shade, 0, 1)


def _objects(net):
    import shapely
    from .roads import _on_ground
    mapped_lamps = []
    lines = [shapely.LineString(r[0]) for r in net.street_lines]
    tree = shapely.STRtree(lines) if lines else None
    for e in net.elements:
        tags = e.get('tags', {})
        if e.get('type') != 'node' or not all(k in e for k in ('lat','lon')) or not _on_ground(tags):
            continue
        kind = tags.get('highway')
        if kind not in ('street_lamp','traffic_signals'):
            kind = tags.get('amenity')
        if kind not in ('street_lamp','traffic_signals','bench','waste_basket'):
            if tags.get('highway') == 'crossing' and tags.get('crossing') == 'traffic_signals':
                kind = 'traffic_signals'
            else:
                continue
        xy = net.to_xy([e])[0]
        direction = np.array([1., 0.])
        if net.street_lines:
            nearest = int(tree.nearest(shapely.Point(xy)))
            _, width, _ = net.street_lines[nearest]
            line = lines[nearest]
            at = line.project(shapely.Point(xy))
            a, b = np.array(line.interpolate(max(0,at-.5)).coords[0]), np.array(line.interpolate(min(line.length,at+.5)).coords[0])
            direction = (b-a) / max(np.linalg.norm(b-a),1e-9)
            if kind == 'traffic_signals' and line.distance(shapely.Point(xy)) < width / 2:
                # a signal node on the centreline: move it to the road's side
                xy = np.array(line.interpolate(at).coords[0]) + np.array([-direction[1],direction[0]]) * (width/2+.45)
        if kind == 'street_lamp':
            mapped_lamps.append(xy)
        yield kind, xy, direction
    # lamps every 28 m along lit sidewalks
    from .roads import _sidewalks, PAVEMENT, KERB, PAVING_JOINT
    pavement = shapely.union_all([net.shapes.get(c,shapely.Polygon()) for c in (PAVEMENT,KERB,PAVING_JOINT)])
    accepted = list(mapped_lamps)
    for xy, width, tags in net.street_lines:
        if tags.get('lit') != 'yes':
            continue
        for _, line, _ in _sidewalks(xy,width,tags):
            if line.geom_type != 'LineString':
                continue
            for at in np.arange(14,line.length-5,28):
                centre = np.array(line.interpolate(at).coords[0])
                if not pavement.contains(shapely.Point(centre)) or any(np.linalg.norm(centre-p)<18 for p in accepted):
                    continue
                a = np.array(line.interpolate(at-.5).coords[0])
                b = np.array(line.interpolate(at+.5).coords[0])
                accepted.append(centre)
                yield 'street_lamp', centre, (b-a)/np.linalg.norm(b-a)


def quads(net, ground):
    from .roads import LIFT_M, KERB_M
    if not hasattr(net, "_street_objects"):
        net._street_objects = list(_objects(net))
    for kind, xy, direction in net._street_objects:
        origin = np.r_[xy, float(ground(xy[None])[0])+LIFT_M+KERB_M]
        def box(bounds, colour=METAL):
            return _box(origin,direction,bounds,colour)
        if kind == 'street_lamp':
            yield from box((-.16,.16,-.16,.16,0,.45))
            yield from box((-.09,.09,-.09,.09,.45,5.8))
            yield from box((-.12,.12,-1.25,.12,5.6,5.82))
            yield from box((-.3,.3,-1.55,-.8,5.48,5.7),(.72,.72,.62))
        elif kind == 'traffic_signals':
            yield from box((-.09,.09,-.09,.09,0,3.4))
            yield from box((-.23,.23,-.22,.17,2.65,3.65),(.12,.14,.14))
            # static lenses
            for z, colour in ((3.38,(.55,.16,.12)),(3.08,(.65,.48,.12)),(2.78,(.13,.38,.23))):
                yield from box((-.13,.13,-.26,-.225,z,z+.2),colour)
        elif kind == 'bench':
            for x in (-.65,.65):
                yield from box((x-.07,x+.07,-.25,.25,0,.48))
            for y in (-.2,0,.2):
                yield from box((-.95,.95,y-.07,y+.07,.45,.56),WOOD)
            for z in (.73,.93):
                yield from box((-.95,.95,.22,.32,z,z+.13),WOOD)
        else:
            yield from box((-.25,.25,-.25,.25,0,.85),(.28,.36,.31))
            yield from box((-.28,.28,-.28,.28,.85,.94),METAL)
