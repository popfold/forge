"""Tidy a phone scan that scan_cut.py has already cut free of its surroundings (run inside Blender).

    blender -b -P scan_clean.py -- CUT.glb OUT.glb [--tris 100000] [--tex 4096] [--turn]

Welds the seams, turns the object so its broad side faces the viewer (front -Y in Blender, +Z in
glTF), puts the base centre at the origin, scales the largest side to 1, thins it to --tris
triangles and shrinks the photo texture to --tex wide. Prints the real size in metres.
--turn: spin it half a turn, to show the other broad side as the front.
"""
import math
import sys

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = argv[0], argv[1]
opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
max_tris, tex = int(opt("--tris", 100000)), int(opt("--tex", 4096))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
o = [o for o in bpy.context.scene.objects if o.type == "MESH"][0]
o.parent = None
for x in [x for x in bpy.context.scene.objects if x.type != "MESH"]:
    bpy.data.objects.remove(x)
bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
bm = bmesh.new(); bm.from_mesh(o.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0008)
bm.to_mesh(o.data); bm.free()
P = np.array([v.co[:] for v in o.data.vertices])

# broad side to the front
xy = P[:, :2] - P[:, :2].mean(0)
w, v = np.linalg.eigh(np.cov(xy.T))
R = Matrix.Rotation(-math.atan2(v[1, 1], v[0, 1]), 4, "Z")

if "--turn" in argv:
    R = Matrix.Rotation(math.pi, 4, "Z") @ R
o.data.transform(R)

P = np.array([v.co[:] for v in o.data.vertices])
lo, hi = P.min(0), P.max(0)
print(f"REAL SIZE m {hi[0] - lo[0]:.3f} wide x {hi[1] - lo[1]:.3f} deep x {hi[2] - lo[2]:.3f} tall")
o.data.transform(Matrix.Scale(1 / max(hi - lo), 4) @ Matrix.Translation(Vector((-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]))))
o.data.update()
t = sum(len(p.vertices) - 2 for p in o.data.polygons)
if t > max_tris:
    m = o.modifiers.new("d", "DECIMATE"); m.ratio = max_tris / t
    bpy.ops.object.modifier_apply(modifier="d")
for img in bpy.data.images:
    if img.size[0] > tex:
        img.scale(tex, int(img.size[1] * tex / img.size[0]))
o.name = "scan"
root = bpy.data.objects.new("scan_root", None); bpy.context.scene.collection.objects.link(root); o.parent = root
bpy.ops.object.select_all(action="SELECT")
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_yup=True)
print(f"EXPORT {out} tris={sum(len(p.vertices) - 2 for p in o.data.polygons)}")
