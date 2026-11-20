"""Raw Hunyuan3D mesh -> clean, coloured, low-poly .glb (run inside Blender).

    blender -b -P blender_lowpoly.py -- RAW.glb TEXTURE.png MASK.png OUT.glb [--tris 2500] [--flat]

Steps: weld + drop loose fragments, voxel remesh (watertight), find which side matches the
source image, decimate to the triangle budget, project the source image on as the texture,
then normalise: front faces -Y (Blender) / +Z (glTF), base centred on the origin, height-ish 1.
"""
import os
import sys

import bmesh
import bpy
import numpy as np

argv = sys.argv[sys.argv.index("--") + 1:]
raw, tex_path, mask_path, out = [os.path.abspath(a) for a in argv[:4]]
tris = int(argv[argv.index("--tris") + 1]) if "--tris" in argv else 2500
flat = "--flat" in argv
palette_mode = "--palette" in argv  # flat colour per part instead of the projected photo
SHARP_FACING = 0.35
PALETTE_K = int(argv[argv.index("--colours") + 1]) if "--colours" in argv else 7  # how squarely a face must look at the source camera to get the sharp image
depth = float(argv[argv.index("--depth") + 1]) if "--depth" in argv else None  # max front-to-back size, for slabs

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=raw)
meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
for o in bpy.context.scene.objects:
    o.select_set(o.type == "MESH")
bpy.context.view_layer.objects.active = meshes[0]
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.view_layer.objects.active
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
obj.data.materials.clear()
me = obj.data

# ---- weld, keep only the big connected pieces ----
bm = bmesh.new(); bm.from_mesh(me)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)
bm.verts.ensure_lookup_table()
seen = np.zeros(len(bm.verts), dtype=np.int32); islands = []
for v in bm.verts:
    if seen[v.index]:
        continue
    stack, comp = [v], []
    seen[v.index] = 1
    while stack:
        a = stack.pop(); comp.append(a)
        for e in a.link_edges:
            b = e.other_vert(a)
            if not seen[b.index]:
                seen[b.index] = 1; stack.append(b)
    islands.append(comp)
big = max(len(c) for c in islands)
junk = [v for c in islands if len(c) < big * 0.03 for v in c]
bmesh.ops.delete(bm, geom=junk, context="VERTS")
bm.to_mesh(me); bm.free()
print(f"islands={len(islands)} kept={sum(1 for c in islands if len(c) >= big * 0.03)}")


def apply_mod(kind, **kw):
    m = obj.modifiers.new(kind, kind)
    for k, v in kw.items():
        setattr(m, k, v)
    bpy.ops.object.modifier_apply(modifier=m.name)


dims = np.array(obj.dimensions)
apply_mod("REMESH", mode="VOXEL", voxel_size=float(dims.max()) / 140, use_smooth_shade=True)

# ---- which horizontal side matches the source image? (silhouette overlap) ----
import bpy_extras  # noqa: F401  (ensures image loading helpers are registered)
mask_img = bpy.data.images.load(mask_path)
W, H = mask_img.size
mask = np.array(mask_img.pixels[:]).reshape(H, W, 4)[::-1, :, 0] > 0.5  # row 0 = top
ys, xs = np.where(mask)
mx0, mx1, my0, my1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
crop = mask[my0:my1, mx0:mx1]
G = 96
gy = (np.arange(G) + 0.5) / G * crop.shape[0]
gx = (np.arange(G) + 0.5) / G * crop.shape[1]
target = crop[gy.astype(int)][:, gx.astype(int)]

verts = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", verts); verts = verts.reshape(-1, 3)


def silhouette(right):
    u = verts @ right; v = verts[:, 2]
    un = (u - u.min()) / (np.ptp(u) + 1e-9); vn = (v.max() - v) / (np.ptp(v) + 1e-9)
    grid = np.zeros((G, G), bool)
    grid[np.clip((vn * G).astype(int), 0, G - 1), np.clip((un * G).astype(int), 0, G - 1)] = True
    aspect = np.ptp(u) / (np.ptp(v) + 1e-9)
    return grid, aspect


img_aspect = crop.shape[1] / crop.shape[0]
best = None
for name, right in {"+X": (1, 0, 0), "-X": (-1, 0, 0), "+Y": (0, 1, 0), "-Y": (0, -1, 0)}.items():
    g, aspect = silhouette(np.array(right, float))
    iou = (g & target).sum() / max(1, (g | target).sum())
    score = iou - abs(np.log(aspect / img_aspect)) * 0.5
    print(f"  side right={name} iou={iou:.3f} aspect={aspect:.2f} (image {img_aspect:.2f}) score={score:.3f}")
    if best is None or score > best[0]:
        best = (score, name, np.array(right, float), iou)
print(f"front: right={best[1]} iou={best[3]:.3f}")

# rotate so the matching view's right vector becomes +X (front then faces -Y)
ang = -np.arctan2(best[2][1], best[2][0])
obj.rotation_euler = (0, 0, ang)
bpy.ops.object.transform_apply(rotation=True)

# ---- low poly ----
cur = sum(len(p.vertices) - 2 for p in me.polygons)
apply_mod("DECIMATE", ratio=min(1.0, tris / cur))
apply_mod("TRIANGULATE")
for p in me.polygons:
    p.use_smooth = not flat

# ---- normalise: base centre at origin, largest dimension = 1 ----
verts = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", verts); verts = verts.reshape(-1, 3)
lo, hi = verts.min(0), verts.max(0)
verts -= np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
verts /= (hi - lo).max()
if depth and np.ptp(verts[:, 1]) > depth:  # flatten things the mesh model made too deep (doors, plates)
    verts[:, 1] *= depth / np.ptp(verts[:, 1])
me.vertices.foreach_set("co", verts.ravel()); me.update()

# ---- projected UVs: the source image, seen from the front ----
lo, hi = verts.min(0), verts.max(0)
un = (verts[:, 0] - lo[0]) / (hi[0] - lo[0]); vn = (hi[2] - verts[:, 2]) / (hi[2] - lo[2])
inset = 0.012  # stay just inside the silhouette so edges don't pick up background
un = inset + un * (1 - 2 * inset); vn = inset + vn * (1 - 2 * inset)
U = (mx0 + un * (mx1 - mx0)) / W
V = 1 - (my0 + vn * (my1 - my0)) / H
uv = me.uv_layers.new(name="UVMap")
loop_v = np.empty(len(me.loops), dtype=np.int32); me.loops.foreach_get("vertex_index", loop_v)
# The texture is an atlas: left half sharp, right half blurred. Faces that look at the camera
# (front = -Y) get the sharp image; side and back faces get the blurred one, so the back doesn't
# repeat the front's details and the sides don't streak.
normals = np.empty(len(me.polygons) * 3); me.polygons.foreach_get("normal", normals); normals = normals.reshape(-1, 3)
facing = -normals[:, 1]
loop_total = np.empty(len(me.polygons), dtype=np.int32); me.polygons.foreach_get("loop_total", loop_total)
loop_facing = np.repeat(facing, loop_total)
# A face only gets the sharp image if the source camera could actually see it: it must face the
# camera clearly (steep faces stretch the image into streaks) and nothing may sit in front of it
# (otherwise e.g. a duvet top gets the pillows painted across it).
from mathutils import Vector
from mathutils.bvhtree import BVHTree
bvh = BVHTree.FromPolygons([tuple(v) for v in verts], [tuple(p.vertices) for p in me.polygons])
centers = np.empty(len(me.polygons) * 3); me.polygons.foreach_get("center", centers); centers = centers.reshape(-1, 3)
start_y = verts[:, 1].min() - 1.0
tol = 0.02  # about 1/50 of the object; voxel surfaces are a little bumpy
visible = np.zeros(len(me.polygons), bool)
for i, c in enumerate(centers):
    if facing[i] <= SHARP_FACING:
        continue
    hit, _, idx, dist = bvh.ray_cast(Vector((c[0], start_y, c[2])), Vector((0, 1, 0)))
    visible[i] = hit is not None and (idx == i or abs((start_y + dist) - c[1]) < tol)
print(f"sharp faces: {visible.sum()} of {len(visible)}")
loop_sharp = np.repeat(visible, loop_total)
lu = U[loop_v] * 0.5 + np.where(loop_sharp, 0.0, 0.5)
uv.data.foreach_set("uv", np.stack([lu, V[loop_v]], 1).ravel())

if palette_mode:
    # ---- flat colours: cluster the source image into a few part colours, give each visible face its
    # part's colour, then spread colours over the hidden faces from their visible neighbours ----
    tex_img = bpy.data.images.load(tex_path); TW, TH = tex_img.size
    px = np.array(tex_img.pixels[:]).reshape(TH, TW, 4)[::-1, :TW // 2, :3]  # sharp half, row 0 = top
    S = px.shape[0]
    mk = mask[(np.arange(S) * H // S)][:, (np.arange(S) * W // S)]
    def feat(rgb):  # colour difference that cares less about light and shade than about hue
        l = rgb @ np.array([0.3, 0.59, 0.11])
        return np.concatenate([(rgb - l[..., None]) * 2.2, l[..., None] * 0.6], -1)
    train = px[mk]; rng = np.random.default_rng(0)
    train = train[rng.choice(len(train), min(len(train), 30000), replace=False)]
    F = feat(train); K = PALETTE_K
    cent = F[rng.choice(len(F), K, replace=False)]
    for _ in range(25):
        lab = ((F[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
        cent = np.array([F[lab == k].mean(0) if (lab == k).any() else cent[k] for k in range(K)])
    lab = ((F[:, None] - cent[None]) ** 2).sum(-1).argmin(1)
    pal = np.array([np.median(train[lab == k], 0) if (lab == k).any() else [0.5, 0.5, 0.5] for k in range(K)])
    labmap = ((feat(px.reshape(-1, 3))[:, None] - cent[None]) ** 2).sum(-1).argmin(1).reshape(S, S)
    print("palette:", " ".join("#%02x%02x%02x" % tuple((c * 255).astype(int)) for c in pal))

    vlab = labmap[np.clip(((1 - V) * S).astype(int), 0, S - 1), np.clip((U * S).astype(int), 0, S - 1)]
    nf = len(me.polygons); face_lab = np.full(nf, -1)
    loop_start = np.empty(nf, dtype=np.int32); me.polygons.foreach_get("loop_start", loop_start)
    for i in np.where(visible)[0]:
        c = centers[i]
        cu_ = (c[0] - lo[0]) / (hi[0] - lo[0]); cv_ = (hi[2] - c[2]) / (hi[2] - lo[2])
        cu_ = (mx0 + (inset + cu_ * (1 - 2 * inset)) * (mx1 - mx0)) / W; cv_ = (my0 + (inset + cv_ * (1 - 2 * inset)) * (my1 - my0)) / H
        votes = [labmap[min(S - 1, int(cv_ * S)), min(S - 1, int(cu_ * S))]] * 2
        votes += [vlab[loop_v[j]] for j in range(loop_start[i], loop_start[i] + loop_total[i])]
        face_lab[i] = np.bincount(votes).argmax()
    # face adjacency over shared edges
    edge_faces = {}
    for p in me.polygons:
        for ek in p.edge_keys:
            edge_faces.setdefault(ek, []).append(p.index)
    nbrs = [[] for _ in range(nf)]
    for fs in edge_faces.values():
        for a in fs:
            nbrs[a].extend(b for b in fs if b != a)
    # one clean-up pass: a lone face that disagrees with all its neighbours takes theirs
    for i in np.where(face_lab >= 0)[0]:
        nl = [face_lab[j] for j in nbrs[i] if face_lab[j] >= 0]
        if nl and face_lab[i] not in nl:
            face_lab[i] = np.bincount(nl).argmax()
    from collections import deque
    q = deque(np.where(face_lab >= 0)[0])
    if not q:
        face_lab[:] = 0
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if face_lab[j] < 0:
                face_lab[j] = face_lab[i]; q.append(j)
    face_lab[face_lab < 0] = 0
    col = me.color_attributes.new("Col", "BYTE_COLOR", "CORNER")
    for p in me.polygons:
        r, g, b = pal[face_lab[p.index]]
        for j in range(p.loop_start, p.loop_start + p.loop_total):
            col.data[j].color_srgb = (r, g, b, 1.0)
    me.uv_layers.remove(me.uv_layers["UVMap"])
    mat = bpy.data.materials.new("palette"); mat.use_nodes = True
    nt = mat.node_tree; bsdf = nt.nodes["Principled BSDF"]
    vc = nt.nodes.new("ShaderNodeVertexColor"); vc.layer_name = "Col"
    nt.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.8; bsdf.inputs["Metallic"].default_value = 0.0
    me.materials.append(mat)
    obj.name = os.path.splitext(os.path.basename(out))[0]
    bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_yup=True)
    print(f"LOWPOLY {out} tris={sum(len(p.vertices) - 2 for p in me.polygons)} bytes={os.path.getsize(out)}")
    raise SystemExit(0)

mat = bpy.data.materials.new("projected"); mat.use_nodes = True
nt = mat.node_tree; bsdf = nt.nodes["Principled BSDF"]
t = nt.nodes.new("ShaderNodeTexImage"); t.image = bpy.data.images.load(tex_path)
nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.85  # fabric and lacquer read better matte in the site
bsdf.inputs["Metallic"].default_value = 0.0
me.materials.append(mat)
obj.name = os.path.splitext(os.path.basename(out))[0]

bpy.ops.export_scene.gltf(filepath=out, export_format="GLB", export_image_format="JPEG", export_jpeg_quality=85, export_yup=True)
print(f"LOWPOLY {out} tris={sum(len(p.vertices) - 2 for p in me.polygons)} bytes={os.path.getsize(out)}")
