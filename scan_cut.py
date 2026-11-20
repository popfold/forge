import bpy,bmesh,sys,numpy as np
from mathutils import Vector, Matrix
a=sys.argv[sys.argv.index("--")+1:]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=a[0])
o=[o for o in bpy.context.scene.objects if o.type=="MESH"][0]
bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active=o
bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
me=o.data
co=np.array([v.co[:] for v in me.vertices])
c=0.02; x0,y0=co[:,0].min(),co[:,1].min()
ix=((co[:,0]-x0)/c).astype(int); iy=((co[:,1]-y0)/c).astype(int)
H=np.full((ix.max()+1,iy.max()+1),-1.0)
np.maximum.at(H,(ix,iy),co[:,2])
mask=H>0.035
seed=np.unravel_index(np.argmax(H),H.shape)
comp=np.zeros_like(mask); st=[seed]
while st:
    i,j=st.pop()
    if i<0 or j<0 or i>=H.shape[0] or j>=H.shape[1] or comp[i,j] or not mask[i,j]: continue
    comp[i,j]=True; st+= [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
def dil(m,n):
    m=m.copy()
    for _ in range(n):
        p=np.pad(m,1); m=p[1:-1,1:-1]|p[2:,1:-1]|p[:-2,1:-1]|p[1:-1,2:]|p[1:-1,:-2]
    return m
foot=dil(comp,1); ring=dil(comp,5)&~dil(comp,3)
inring=ring[ix,iy]
sheet=float(np.median(co[inring,2]))
print("cells",comp.sum(),"sheet z",round(sheet,4), "ring pct", np.percentile(co[inring,2],[10,50,90]).round(4))
cut=sheet+float(a[2])
keep=foot[ix,iy]&(co[:,2]>cut)
bm=bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
bmesh.ops.delete(bm,geom=[v for v in bm.verts if not keep[v.index]],context="VERTS")
# drop small floating pieces
seen=set(); parts=[]
for v in bm.verts:
    if v in seen: continue
    st=[v]; comp_=[]
    seen.add(v)
    while st:
        x=st.pop(); comp_.append(x)
        for e in x.link_edges:
            w=e.other_vert(x)
            if w not in seen: seen.add(w); st.append(w)
    parts.append(comp_)
parts.sort(key=len,reverse=True)
print("pieces",[len(p) for p in parts[:8]])
big=len(parts[0])
bmesh.ops.delete(bm,geom=[v for p in parts if len(p)<0.02*big for v in p],context="VERTS")
# flatten the bottom edge onto the cut plane and cap it
bnd=[e for e in bm.edges if e.is_boundary]
for e in bnd:
    for v in e.verts:
        if v.co.z<cut+0.012: v.co.z=cut
r=bmesh.ops.holes_fill(bm,edges=bnd,sides=0)
print("filled",len(r["faces"]))
bm.to_mesh(me); bm.free()
# base centre to origin, bottom at z=0
co=np.array([v.co[:] for v in me.vertices])
lo,hi=co.min(0),co.max(0)
me.transform(Matrix.Translation(Vector((-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2]))))
print("dims m",(hi-lo).round(3),"verts",len(me.vertices),"faces",len(me.polygons))
bpy.ops.export_scene.gltf(filepath=a[1],export_format="GLB")
