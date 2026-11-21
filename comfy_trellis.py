"""Image -> textured low-poly mesh (.glb) with TRELLIS.2 in ComfyUI: shape and colour on every side.

    python comfy_trellis.py IMAGE.png [--faces 3000] [--texture 1024] [--seed 42]

The picture is cut out (BiRefNet), TRELLIS.2 builds a sparse structure, then the shape, then its colours;
ComfyUI remeshes it into one solid, decimates it to --faces, unwraps it and bakes the colours into a
--texture square base colour map (plus metallic and roughness). The result is saved as
<out>/meshes/<name>.glb (see config.py), with the time and peak VRAM.

Needs ComfyUI 0.46 or newer and these files (README.md, "TRELLIS.2"):
  models/diffusion_models/trellis_2_int8_convrot.safetensors
  models/vae/trellis_2_shape_vae_bf16.safetensors, models/vae/trellis_2_texture_vae_bf16.safetensors
  models/clip_vision/dino_v3_L_naf_fp32.safetensors
  models/background_removal/birefnet.safetensors
The graph follows ComfyUI's own TRELLIS.2 template (its sampler settings and guidance), built in code.
"""
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import config
from comfy_3d import upload
from comfy_run import SERVER, VramWatch, api

OUT = config.OUT / "meshes"
FILES = {
    "unet": "trellis_2_int8_convrot.safetensors",
    "shape_vae": "trellis_2_shape_vae_bf16.safetensors",
    "texture_vae": "trellis_2_texture_vae_bf16.safetensors",
    "vision": "dino_v3_L_naf_fp32.safetensors",
    "cutout": "birefnet.safetensors",
}


def workflow(image_name, name, faces=3000, texture=1024, seed=42, background="#000000"):
    n = lambda cls, **inputs: {"class_type": cls, "inputs": inputs}  # noqa: E731
    ks = lambda model, cond, seed, steps, cfg, scheduler="normal": n(  # noqa: E731
        "KSampler", model=model, positive=[cond, 0], negative=[cond, 1], latent_image=[cond, 2], seed=seed, steps=steps,
        cfg=cfg, sampler_name="euler", scheduler=scheduler, denoise=1.0)
    return {
        "img": n("LoadImage", image=image_name),
        "bgm": n("LoadBackgroundRemovalModel", bg_removal_name=FILES["cutout"]),
        "mask": n("RemoveBackground", bg_removal_model=["bgm", 0], image=["img", 0]),
        "crop": n("ImageCropToMask", images=["img", 0], masks=["mask", 0], width=1024, height=1024, pad_factor=1.0,
                  grow_mask=0, background=background),
        "vis": n("CLIPVisionLoader", clip_name=FILES["vision"]),
        "cond": n("Trellis2Conditioning", clip_vision_model=["vis", 0], image=["crop", 0]),
        "unet": n("UNETLoader", unet_name=FILES["unet"], weight_dtype="default"),
        "svae": n("VAELoader", vae_name=FILES["shape_vae"]),
        "tvae": n("VAELoader", vae_name=FILES["texture_vae"]),
        # guidance as in ComfyUI's template, which matches the original pipeline
        "m_struct": n("ModelSamplingSD3", model=["rs_struct", 0], shift=5.0),
        "rs_struct": n("RescaleCFG", model=["co_struct", 0], multiplier=0.7),
        "co_struct": n("CFGOverride", model=["unet", 0], cfg=1.0, start_percent=0.667, end_percent=1.0),
        "m_shape": n("RescaleCFG", model=["co_shape", 0], multiplier=0.5),
        "co_shape": n("CFGOverride", model=["unet", 0], cfg=1.0, start_percent=0.769, end_percent=1.0),
        # 1. sparse structure
        "empty": n("EmptyTrellis2LatentStructure", batch_size=1),
        "s_struct": n("KSampler", model=["m_struct", 0], positive=["cond", 0], negative=["cond", 1], latent_image=["empty", 0],
                      seed=seed + 14, steps=12, cfg=7.5, sampler_name="euler", scheduler="normal", denoise=1.0),
        "voxel": n("VaeDecodeStructureTrellis2", samples=["s_struct", 0], vae=["svae", 0], resolution="32"),
        # 2. shape, then shape at a finer resolution
        "shape": n("Trellis2ShapeStage", positive=["cond", 0], negative=["cond", 1], voxel=["voxel", 0]),
        "s_shape": ks(["m_shape", 0], "shape", seed, 20, 7.5),
        "up": n("Trellis2UpsampleStage", positive=["shape", 0], negative=["shape", 1], shape_latent=["s_shape", 0],
                # the template's 1536 takes ~45 s a step on a 12 GB card (memory spills); 1024 is plenty for a
                # prop decimated to a few thousand faces
                vae=["svae", 0], target_resolution=1024),
        "s_up": ks(["m_shape", 0], "up", seed, 12, 7.5, "simple"),
        "mesh": n("VaeDecodeShapeTrellis", samples=["s_up", 0], vae=["svae", 0]),
        # 3. colours
        "tex": n("Trellis2TextureStage", positive=["up", 0], negative=["up", 1], shape_latent=["s_up", 0]),
        "s_tex": ks(["unet", 0], "tex", seed + 1, 12, 1.0),
        "colours": n("VaeDecodeTextureTrellis", samples=["s_tex", 0], vae=["tvae", 0], shape_subdivides=["mesh", 1]),
        # 4. one solid, low poly, unwrapped, colours baked into a texture
        "remesh": n("RemeshMesh", mesh=["mesh", 0], resolution=768, **{"sign_mode": "solid", "sign_mode.fill": 20.0,
                    "sign_mode.qef": False}, band=1.0, project_back=0.0, fix_poles=False, smooth_iters=0,
                    drop_small_components=0.01, precluster_max_verts=20000000),
        "low": n("DecimateMesh", mesh=["remesh", 0], target_face_count=int(faces), placement_mode="midpoint"),
        "smooth": n("MeshSmoothNormals", mesh=["low", 0], crease_angle=180.0),
        "uv": n("UnwrapMesh", mesh=["smooth", 0], segmenter="pec", resolution=int(texture), padding=1, weld_distance=0.0002),
        "bake": n("BakeTextureFromVoxel", mesh=["uv", 0], voxel_colors=["colours", 0], texture_size=int(texture),
                  reference_mesh=["mesh", 0]),
        "painted": n("ApplyTextureToMesh", mesh=["uv", 0], base_color=["bake", 0], metallic=["bake", 1], roughness=["bake", 2]),
        "save": n("SaveGLB", mesh=["painted", 0], filename_prefix="popfold3d/" + name),
    }


def run(path: Path, faces=3000, texture=1024, seed=42, timeout=5400, background="#000000"):
    name = path.stem
    image_name = upload(path)
    watch = VramWatch(); watch.start()
    t0 = time.time()
    r = api("/prompt", {"prompt": workflow(image_name, name, faces, texture, seed, background), "client_id": str(uuid.uuid4())})
    if r.get("node_errors"):
        raise RuntimeError(f"{name}: ComfyUI refused the graph: {json.dumps(r['node_errors'])[:1500]}")
    pid = r["prompt_id"]
    while True:
        h = api(f"/history/{pid}")
        if pid in h:
            entry = h[pid]; break
        if time.time() - t0 > timeout:
            raise RuntimeError(f"{name}: not done after {timeout} s")
        time.sleep(2)
    secs = time.time() - t0; watch.stop = True
    if entry.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"{name} failed: {json.dumps(entry.get('status'))[:1500]}")
    OUT.mkdir(parents=True, exist_ok=True)
    saved = ""
    for node in entry["outputs"].values():
        for items in node.values():
            for f in items if isinstance(items, list) else []:
                if isinstance(f, dict) and str(f.get("filename", "")).endswith(".glb"):
                    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
                    dest = OUT / f"{name}.glb"
                    dest.write_bytes(urllib.request.urlopen(f"{SERVER}/view?{q}", timeout=300).read())
                    saved = str(dest)
    if not saved:
        raise RuntimeError(f"{name}: ComfyUI saved no .glb")
    return {"name": name, "seconds": round(secs, 1), "peak_vram_mib": watch.peak, "file": saved}


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    values = {a[a.index(k) + 1] for k in ("--faces", "--texture", "--seed") if k in a}
    for p in [x for x in a if not x.startswith("--") and x not in values]:
        r = run(Path(p), opt("--faces", 3000), opt("--texture", 1024), opt("--seed", 42))
        print(f"{r['name']:<24} {r['seconds']:>6}s  peak {r['peak_vram_mib']} MiB  -> {r['file']}", flush=True)
