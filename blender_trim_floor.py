"""Cut away a floor slab a model was made standing on (run inside Blender).

    blender -b -P blender_trim_floor.py -- IN.glb OUT.glb

Some image-to-3D services (Hunyuan3D 2.1's Space, often) build the floor under the object as a flat square
plate, and the object's feet often poke through it. Seen from the side, the plate is a thin run of slices, low
down, much wider than the object above. It deletes the faces low down outside the object's footprint (most of the
plate, warped or not), and, under the object, the flat faces from the plate's bottom up to its top, then any
small loose crumbs left there; the feet, whose sides are upright, stay. A model without a plate is written out unchanged.
Prints "TRIM <out> slab=<yes|no> band=<plate's heights, as fractions> faces=<before>-><after>".
"""
import os
import sys

import bmesh
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = [os.path.abspath(a) for a in argv[:2]]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in bpy.context.scene.objects:
    o.select_set(o.type == "MESH")
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

v = np.array([p.co[:] for p in obj.data.vertices])
z0, z1 = v[:, 2].min(), v[:, 2].max()
h = z1 - z0


def area(lo, hi):
    """The footprint (bounding rectangle) of the vertices between two heights, as fractions of the height."""
    s = v[(v[:, 2] >= z0 + lo * h) & (v[:, 2] < z0 + hi * h)]
    if len(s) < 3:
        return 0.0
    return float(np.ptp(s[:, 0]) * np.ptp(s[:, 1]))


step = 0.0025
slices = [(k, area(k, k + step)) for k in np.arange(0, 0.3, step)]
widest = max(a for _k, a in slices)
before = len(obj.data.polygons)
band = None
if widest > 1.3 * area(0.3, 1.01):  # some slice low down is much wider than the object above: a plate
    wide = [k for k, a in slices if a > 0.6 * widest]
    lo = hi = wide[0]
    for k in wide[1:]:  # the plate is one run of wide slices
        if k - hi > 2.5 * step:
            break
        hi = k
    band = (lo - step, hi + 2 * step)

if band is not None:
    zl, zh = z0 + band[0] * h, z0 + band[1] * h
    # the object's own footprint, from just above the plate, so a wide base on it (rocks, a hill) counts as the
    # object; higher up only while the slice is still plate-wide (a plate is rarely quite flat)
    for lift in (0.02, 0.04, 0.06, 0.1):
        if area(band[1] + lift, band[1] + lift + 0.02) < 0.5 * widest:
            break
    s = v[v[:, 2] > zh + lift * h]
    (x0, y0), (x1, y1) = s[:, :2].min(0), s[:, :2].max(0)
    mx, my = 0.03 * (x1 - x0), 0.03 * (y1 - y0)
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.normal_update()

    def outside(c):
        return not (x0 - mx <= c.x <= x1 + mx and y0 - my <= c.y <= y1 + my)

    # the plate's top: the highest flat face of it outside the object (a plate has a thickness)
    tops = [f.calc_center_median().z for f in bm.faces if abs(f.normal.z) > 0.7
            and outside(f.calc_center_median()) and f.calc_center_median().z < zh + lift * h]
    zt = max(tops, default=zh)

    def plate(f):
        c = f.calc_center_median()
        if outside(c):  # the plate, however warped, up to where the footprint was taken
            return c.z < zh + lift * h
        # under the object: the plate's top and bottom, a little rough, so not quite flat
        return zl - 0.01 * h <= c.z <= zt + 0.005 * h and abs(f.normal.z) > 0.3
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if plate(f)], context="FACES")
    # crumbs: small loose pieces left in or under the old plate
    bm.verts.ensure_lookup_table()
    seen, crumbs = set(), []
    for vert in bm.verts:
        if vert in seen:
            continue
        part, todo = [], [vert]
        seen.add(vert)
        while todo:
            x = todo.pop()
            part.append(x)
            for e in x.link_edges:
                y = e.other_vert(x)
                if y not in seen:
                    seen.add(y)
                    todo.append(y)
        if len(part) < 0.02 * len(bm.verts) and max(p.co.z for p in part) < zh + 0.02 * h:
            crumbs += part
    bmesh.ops.delete(bm, geom=crumbs, context="VERTS")
    bm.verts.ensure_lookup_table()
    bm.to_mesh(obj.data)
    bm.free()

bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
print(f"TRIM {out} slab={'yes' if band else 'no'} band={band and tuple(round(float(b), 4) for b in band)} "
      f"faces={before}->{len(obj.data.polygons)}", flush=True)
