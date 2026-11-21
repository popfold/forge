"""A finished textured model -> Popfold's prop convention, keeping its own texture (run inside Blender).

    blender -b -P blender_fit.py -- IN.glb MASK.png OUT.glb [--tris 3000]

For models that come with their colours already (TRELLIS.2): finds which horizontal side matches the
picture's cut-out (MASK.png, white = object), turns that side to the front, reduces it to --tris triangles
with Blender's collapse decimation (which keeps the UVs, so the texture still fits), then normalises: front faces -Y
(Blender) / +Z (glTF), base centred on the origin, largest dimension 1. Textures are written as JPEG.
Prints "FIT <out> front=<side> iou=<overlap> tris=<n> bytes=<n>".
"""
import os
import sys

import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
src, mask_path, out = [os.path.abspath(a) for a in argv[:3]]
tris = int(argv[argv.index("--tris") + 1]) if "--tris" in argv else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in bpy.context.scene.objects:
    o.select_set(o.type == "MESH")
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True)
bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = obj.data

# ---- low poly: Blender's collapse keeps thin parts far better than a quick midpoint decimation ----
if tris:
    cur = sum(len(p.vertices) - 2 for p in me.polygons)
    if cur > tris:
        m = obj.modifiers.new("dec", "DECIMATE"); m.ratio = tris / cur; m.use_collapse_triangulate = True
        bpy.ops.object.modifier_apply(modifier=m.name)

# ---- which horizontal side matches the picture? (silhouette overlap, as in blender_lowpoly.py) ----
mask_img = bpy.data.images.load(mask_path)
W, H = mask_img.size
mask = np.array(mask_img.pixels[:]).reshape(H, W, -1)[::-1, :, 0] > 0.5
ys, xs = np.where(mask)
crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
G = 96
target = crop[((np.arange(G) + 0.5) / G * crop.shape[0]).astype(int)][:, ((np.arange(G) + 0.5) / G * crop.shape[1]).astype(int)]
verts = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", verts); verts = verts.reshape(-1, 3)
# the silhouette is every triangle filled in: sampling only vertices left a 1000-triangle sideboard's big flat
# faces empty, and a good model scored 0.17
me.calc_loop_triangles()
tri = np.empty(len(me.loop_triangles) * 3, dtype=np.int64); me.loop_triangles.foreach_get("vertices", tri)
tri = tri.reshape(-1, 3)
cy, cx = np.mgrid[0:G, 0:G] + 0.5


def silhouette(u, v):
    """Rasterise the triangles, projected to (u across, v down), onto a G x G grid fitted to their bounds."""
    x = (u - u.min()) / (np.ptp(u) + 1e-9) * G; y = (v.max() - v) / (np.ptp(v) + 1e-9) * G
    g = np.zeros((G, G), bool)
    for a, b, c in tri:
        xs, ys = x[[a, b, c]], y[[a, b, c]]
        x0, x1 = int(max(0, xs.min())), int(min(G - 1, xs.max())); y0, y1 = int(max(0, ys.min())), int(min(G - 1, ys.max()))
        px, py = cx[y0:y1 + 1, x0:x1 + 1], cy[y0:y1 + 1, x0:x1 + 1]
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-12:
            g[np.clip(ys.astype(int), 0, G - 1), np.clip(xs.astype(int), 0, G - 1)] = True  # edge-on: mark its corners
            continue
        l1 = ((ys[1] - ys[2]) * (px - xs[2]) + (xs[2] - xs[1]) * (py - ys[2])) / d
        l2 = ((ys[2] - ys[0]) * (px - xs[2]) + (xs[0] - xs[2]) * (py - ys[2])) / d
        g[y0:y1 + 1, x0:x1 + 1] |= (l1 >= -0.02) & (l2 >= -0.02) & (1 - l1 - l2 >= -0.02)
    return g


img_aspect = crop.shape[1] / crop.shape[0]
best = None
for name, right in {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0)}.items():
    r = np.array(right, float)
    u = verts @ r; v = verts[:, 2]
    g = silhouette(u, v)
    # a side and its opposite give mirror-image silhouettes; the picture's asymmetry tells them apart
    iou = (g & target).sum() / max(1, (g | target).sum())
    aspect = np.ptp(u) / (np.ptp(v) + 1e-9)
    score = iou - abs(np.log(aspect / img_aspect)) * 0.5
    print(f"  side right={name} iou={iou:.3f} aspect={aspect:.2f} (image {img_aspect:.2f}) score={score:.3f}")
    if best is None or score > best[0]:
        best = (score, name, r, iou)

ang = -np.arctan2(best[2][1], best[2][0])
obj.rotation_euler = (0, 0, ang)
bpy.ops.object.transform_apply(rotation=True)

# ---- normalise: base centre at origin, largest dimension = 1 ----
verts = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", verts); verts = verts.reshape(-1, 3)
lo, hi = verts.min(0), verts.max(0)
verts -= np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
verts /= (hi - lo).max()
me.vertices.foreach_set("co", verts.ravel()); me.update()
obj.name = os.path.splitext(os.path.basename(out))[0]
for m in me.materials:  # glossy bakes read as plastic in the site; keep them matte like the other props
    if m and m.use_nodes and "Principled BSDF" in m.node_tree.nodes:
        b = m.node_tree.nodes["Principled BSDF"]
        if not b.inputs["Roughness"].links:
            b.inputs["Roughness"].default_value = max(0.6, b.inputs["Roughness"].default_value)

os.makedirs(os.path.dirname(out), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_image_format="JPEG", export_jpeg_quality=85, export_yup=True)
tris = sum(len(p.vertices) - 2 for p in me.polygons)
print(f"FIT {out} front={best[1]} iou={best[3]:.3f} tris={tris} bytes={os.path.getsize(out)}")
