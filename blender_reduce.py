"""Reduce a dense model to a triangle budget, shape only (run inside Blender).

    blender -b -P blender_reduce.py -- IN.glb OUT.glb [--tris 3000]

A glTF splits every vertex along its UV seams, so a textured model from a service arrives as hundreds of separate
islands. Blender's collapse cannot merge across them: it stalls far above the budget and creases the surface.
This welds the seams first, then collapses (again, if it stops short), and drops the UVs and materials, which no
longer fit. Give it colours again with blender_transfer.py, baking them from the dense model. Orient and
normalise the dense model first (blender_fit.py without --tris), so the two line up.
Prints "REDUCE <out> tris=<n>".
"""
import os
import sys

import bmesh
import bpy

argv = sys.argv[sys.argv.index("--") + 1:]
src, out = [os.path.abspath(a) for a in argv[:2]]
tris = int(argv[argv.index("--tris") + 1]) if "--tris" in argv else 3000

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in bpy.context.scene.objects:
    o.select_set(o.type == "MESH")
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
me = obj.data

bm = bmesh.new()
bm.from_mesh(me)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
bm.to_mesh(me)
bm.free()


def count():
    return sum(len(p.vertices) - 2 for p in me.polygons)


for _ in range(6):
    if count() <= tris * 1.03:
        break
    m = obj.modifiers.new("dec", "DECIMATE")
    m.ratio = tris / count()
    m.use_collapse_triangulate = True
    bpy.ops.object.modifier_apply(modifier=m.name)

while me.uv_layers:
    me.uv_layers.remove(me.uv_layers[0])
me.materials.clear()
for p in me.polygons:
    p.use_smooth = True
bpy.ops.object.select_all(action="DESELECT")
obj.select_set(True)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", use_selection=True)
print(f"REDUCE {out} tris={count()}", flush=True)
