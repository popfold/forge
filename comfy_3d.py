"""Image -> 3D mesh (.glb) with ComfyUI's built-in Hunyuan3D 2 / 2.1 nodes (shape only, no texture).

    python comfy_3d.py [--model hunyuan21] IMAGE.png [IMAGE2.png ...]

Each image is uploaded to the ComfyUI server, turned into a mesh, and saved as
<out>/meshes/<name>.glb (see config.py). Timing and peak VRAM are appended to
<out>/results-3d.csv.
Works best with a single object on a plain white background.
"""

import csv
import json
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import config
from comfy_run import SERVER, VramWatch, api

OUT = config.OUT / "meshes"
# the same nodes run both; 2.1 follows ComfyUI's own 2.1 template (a finer latent, a centred crop)
MODELS = {
    "hunyuan": {"ckpt": "hunyuan3d-dit-v2.safetensors", "resolution": 3072, "crop": "none"},
    "hunyuan21": {"ckpt": "hunyuan_3d_v2.1.safetensors", "resolution": 4096, "crop": "center"},
}
CKPT = MODELS["hunyuan"]["ckpt"]


def upload(path: Path) -> str:
    boundary = uuid.uuid4().hex
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{path.name}\"\r\n"
        "Content-Type: image/png\r\n\r\n"
    ).encode() + path.read_bytes() + (
        f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(SERVER + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())["name"]


def workflow(image_name: str, name: str, seed: int = 1, model: str = "hunyuan"):
    m = MODELS[model]
    return {
        "1": {"class_type": "ImageOnlyCheckpointLoader", "inputs": {"ckpt_name": m["ckpt"]}},
        "2": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "3": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["1", 1], "image": ["2", 0], "crop": m["crop"]}},
        "4": {"class_type": "Hunyuan3Dv2Conditioning", "inputs": {"clip_vision_output": ["3", 0]}},
        "5": {"class_type": "EmptyLatentHunyuan3Dv2", "inputs": {"resolution": m["resolution"], "batch_size": 1}},
        "6": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 1.0}},
        "7": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["5", 0],
            "seed": seed, "steps": 30, "cfg": 5.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "8": {"class_type": "VAEDecodeHunyuan3D", "inputs": {"samples": ["7", 0], "vae": ["1", 2], "num_chunks": 8000, "octree_resolution": 256}},
        "9": {"class_type": "VoxelToMesh", "inputs": {"voxel": ["8", 0], "algorithm": "surface net", "threshold": 0.6}},
        "10": {"class_type": "SaveGLB", "inputs": {"mesh": ["9", 0], "filename_prefix": "popfold3d/" + name}},
    }


def run(path: Path, model: str = "hunyuan"):
    name = path.stem
    image_name = upload(path)
    watch = VramWatch()
    watch.start()
    t0 = time.time()
    pid = api("/prompt", {"prompt": workflow(image_name, name, model=model), "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = api(f"/history/{pid}")
        if pid in h:
            entry = h[pid]
            break
        time.sleep(1)
    secs = time.time() - t0
    watch.stop = True
    if entry.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f"{name} failed: {json.dumps(entry.get('status'))[:1200]}")
    saved = ""
    for node in entry["outputs"].values():
        for items in node.values():
            for f in items if isinstance(items, list) else []:
                if isinstance(f, dict) and str(f.get("filename", "")).endswith(".glb"):
                    q = urllib.parse.urlencode({"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
                    dest = OUT / f"{name}.glb"
                    dest.write_bytes(urllib.request.urlopen(f"{SERVER}/view?{q}", timeout=120).read())
                    saved = str(dest)
    return {"name": name, "seconds": round(secs, 1), "peak_vram_mib": watch.peak, "file": saved}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT.parent / "results-3d.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "seconds", "peak_vram_mib", "file"])
        if new:
            w.writeheader()
        args = sys.argv[1:]
        model = "hunyuan"
        if "--model" in args:
            i = args.index("--model"); model = args[i + 1]; del args[i:i + 2]
        for p in args:
            r = run(Path(p), model)
            w.writerow(r)
            f.flush()
            print(f"{r['name']:<24} {r['seconds']:>6}s  peak {r['peak_vram_mib']} MiB  -> {r['file'] or 'NO GLB FOUND'}", flush=True)


if __name__ == "__main__":
    main()
