"""Give one model's shape another model's colours (run inside Blender).

    blender -b -P blender_transfer.py -- SHAPE.glb COLOURS.glb OUT.glb [--size 1024] [--reach 0.06]

Both models are props in Popfold's convention (front -Y, base on the origin, largest side 1), made from the same
picture, so they already sit nearly on top of each other; their bounding boxes are matched exactly first. The
shape gets fresh UVs (Smart UV Project: a front projection maps the back onto the front), and Cycles bakes the
colour model's base colour onto it, looking along each face's normal up to --reach in both directions.

Why: Hunyuan3D 2.1 makes the cleaner shape but no colours; TRELLIS.2 colours every side. This keeps the first's
shape with the second's colours.
"""
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
shape_path, colour_path, out_path = argv[:3]
size = int(argv[argv.index("--size") + 1]) if "--size" in argv else 1024
reach = float(argv[argv.index("--reach") + 1]) if "--reach" in argv else 0.06

bpy.ops.wm.read_factory_settings(use_empty=True)


def load(path):
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=path)
    meshes = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    o = bpy.context.view_layer.objects.active
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return o


def box(o):
    pts = [o.matrix_world @ v.co for v in o.data.vertices]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


shape = load(shape_path)
colour = load(colour_path)

# match the colour model's box to the shape's, axis by axis
(slo, shi), (clo, chi) = box(shape), box(colour)
for o in (colour,):
    for v in o.data.vertices:
        for k in range(3):
            span = (chi[k] - clo[k]) or 1.0
            v.co[k] = slo[k] + (v.co[k] - clo[k]) / span * (shi[k] - slo[k])

# fresh UVs on the shape, and an image to bake into
bpy.ops.object.select_all(action="DESELECT")
shape.select_set(True)
bpy.context.view_layer.objects.active = shape
while shape.data.uv_layers:
    shape.data.uv_layers.remove(shape.data.uv_layers[0])
shape.data.uv_layers.new(name="UVMap")
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.uv.smart_project(angle_limit=1.15, island_margin=0.01)
bpy.ops.object.mode_set(mode="OBJECT")

img = bpy.data.images.new("baked", size, size)
mat = bpy.data.materials.new("baked")
mat.use_nodes = True
nodes = mat.node_tree.nodes
bsdf = nodes.get("Principled BSDF")
tex = nodes.new("ShaderNodeTexImage")
tex.image = img
mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.5
nodes.active = tex
shape.data.materials.clear()
shape.data.materials.append(mat)

# bake: the colour model selected, the shape active
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.samples = 1
scene.cycles.device = "CPU"
bake = scene.render.bake
bake.use_selected_to_active = True
bake.cage_extrusion = reach
bake.max_ray_distance = reach * 2
bake.margin = 8
bake.use_pass_direct = False
bake.use_pass_indirect = False
bake.use_pass_color = True
colour.select_set(True)
shape.select_set(True)
bpy.context.view_layer.objects.active = shape
bpy.ops.object.bake(type="DIFFUSE")
img.pack()

bpy.data.objects.remove(colour)
bpy.ops.object.select_all(action="DESELECT")
shape.select_set(True)
bpy.ops.export_scene.gltf(filepath=out_path, export_format="GLB", use_selection=True)
print(f"transfer: {len(shape.data.polygons)} faces, {size}px colours -> {out_path}", flush=True)
