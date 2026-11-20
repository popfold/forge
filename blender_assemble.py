"""Assemble pipeline parts into one animated .glb (run inside Blender).

    blender -b -P blender_assemble.py -- SPEC.json OUT.glb [--blend OUT.blend]

SPEC: {"name", "fps"?: 24, "parts": [{
    "name", "file",                      # a low-poly .glb from the pipeline (origin = base centre, front = -Y)
    "parent"?: other part name,
    "scale"?: 1.0,                       # size relative to its own normalised size (largest side = 1)
    "at"?: [fx, fy, fz],                 # where its origin goes, as fractions of the PARENT's bounding box
                                         #   x: 0 left..1 right, y: 0 front..1 back, z: 0 bottom..1 top
    "offset"?: [x, y, z],                # extra shift in Blender units
    "rot"?: [x, y, z] degrees,           # rest rotation about the part's origin
    "origin"?: [fx, fy, fz],             # move the part's own pivot (fractions of its box) before placing
    "anim"?: {"clip": "Flip", "keys": [{"f": 1, "rot": [..]} | {"f": 1, "loc": [..]} | {"f":1,"scale":[..]}]}
}]}
Each animated part gets its own action, pushed to an NLA track named after the clip, so the
exported glTF has one animation per clip name that three.js can play.
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
spec_path, out = os.path.abspath(argv[0]), os.path.abspath(argv[1])
blend = os.path.abspath(argv[argv.index("--blend") + 1]) if "--blend" in argv else None
spec = json.load(open(spec_path, encoding="utf-8"))
base = os.path.dirname(spec_path)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = spec.get("fps", 24)
objs = {}


def bbox(o):
    pts = [Vector(c) for c in o.bound_box]
    return Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts))), \
           Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))


def frac(o, f):
    lo, hi = bbox(o)
    return Vector((lo.x + (hi.x - lo.x) * f[0], lo.y + (hi.y - lo.y) * f[1], lo.z + (hi.z - lo.z) * f[2]))


last = 1
for part in spec["parts"]:
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=os.path.join(base, part["file"]))
    o = next(x for x in set(bpy.data.objects) - before if x.type == "MESH")
    for extra in set(bpy.data.objects) - before - {o}:
        bpy.data.objects.remove(extra)
    o.name = part["name"]
    bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
    o.parent = None
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    s = part.get("scale", 1.0)
    o.data.transform(__import__("mathutils").Matrix.Scale(s, 4))
    if "origin" in part:  # shift geometry so the requested point becomes the pivot
        p = frac(o, part["origin"])
        o.data.transform(__import__("mathutils").Matrix.Translation(-p))
    o.data.update()
    parent = objs.get(part.get("parent"))
    if parent:
        o.parent = parent
        o.location = frac(parent, part.get("at", [0.5, 0.5, 1.0]))
    o.location += Vector(part.get("offset", [0, 0, 0]))
    o.rotation_mode = "XYZ"
    o.rotation_euler = [math.radians(a) for a in part.get("rot", [0, 0, 0])]
    objs[part["name"]] = o

    anim = part.get("anim")
    if anim:
        rest = (o.location.copy(), o.rotation_euler.copy(), o.scale.copy())
        for k in anim["keys"]:
            f = k["f"]; last = max(last, f)
            if "rot" in k:
                o.rotation_euler = [math.radians(a) for a in k["rot"]]; o.keyframe_insert("rotation_euler", frame=f)
            if "loc" in k:
                o.location = rest[0] + Vector(k["loc"]); o.keyframe_insert("location", frame=f)
            if "scale" in k:
                o.scale = k["scale"]; o.keyframe_insert("scale", frame=f)
        act = o.animation_data.action
        act.name = f"{anim['clip']}_{o.name}"
        track = o.animation_data.nla_tracks.new(); track.name = anim["clip"]
        track.strips.new(anim["clip"], 1, act)
        o.animation_data.action = None
        scene.frame_set(1)

scene.frame_start, scene.frame_end = 1, last
if blend:
    bpy.ops.wm.save_as_mainfile(filepath=blend)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_image_format="JPEG", export_jpeg_quality=85,
                          export_yup=True, export_animations=True, export_animation_mode="NLA_TRACKS",
                          export_merge_animation="NLA_TRACK")
tris = sum(len(p.vertices) - 2 for o in objs.values() for p in o.data.polygons)
print(f"ANIM {out} parts={len(objs)} tris={tris} frames={last} bytes={os.path.getsize(out)}")
