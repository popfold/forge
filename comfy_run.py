"""Run a batch of text-to-image jobs through a local ComfyUI server's HTTP API.

    python comfy_run.py jobs.json            # runs every job in the file
    python comfy_run.py jobs.json pattern-1  # runs only the named job(s)

Each job: {"name", "prompt", "negative"?, "model"?: "sdxl"|"flux"|"chroma", "width"?, "height"?,
           "steps"?, "cfg"?, "seed"?, "tile"?: bool}
Images go to <out>/images/ (see config.py), with a results.csv log of time and peak VRAM.
"""

import csv
import json
import random
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import config

SERVER = config.COMFY_URL
OUT = config.OUT / "images"
MODELS = {
    "sdxl": {"ckpt": "sd_xl_base_1.0.safetensors", "steps": 30, "cfg": 6.5, "sampler": "dpmpp_2m", "scheduler": "karras"},
    "flux": {"ckpt": "flux1-schnell-fp8.safetensors", "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    # Chroma: FLUX-based, trained without FLUX's filtering of adult shapes (asked for novelty-toy tentacle tips,
    # FLUX schnell drew mushroom caps). Separate files: the model, a T5 text encoder, FLUX's own picture decoder
    "chroma": {"unet": "Chroma1-HD-fp8mixed.safetensors", "clip": "t5xxl_fp8_e4m3fn_scaled.safetensors",
               "vae": "ae.safetensors", "steps": 26, "cfg": 4.0, "sampler": "euler", "scheduler": "beta"},
}
DEFAULT_NEG = "text, watermark, logo, signature, blurry, lowres, jpeg artifacts, deformed, extra fingers"


def api(path, data=None):
    req = urllib.request.Request(SERVER + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def workflow(job):
    m = MODELS[job.get("model", "sdxl")]
    seed = job.get("seed", random.randint(0, 2**31))
    if "unet" in m:  # model, text encoder and decoder in separate files
        wf = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": m["unet"], "weight_dtype": "default"}},
              "10": {"class_type": "CLIPLoader", "inputs": {"clip_name": m["clip"], "type": job.get("model")}},
              "11": {"class_type": "VAELoader", "inputs": {"vae_name": m["vae"]}}}
        clip_ref, model_ref, vae_ref = ["10", 0], ["1", 0], ["11", 0]
    else:
        wf = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": m["ckpt"]}}}
        clip_ref, model_ref, vae_ref = ["1", 1], ["1", 0], ["1", 2]
    wf |= {
        "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": job["prompt"]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": clip_ref, "text": job.get("negative", DEFAULT_NEG)}},
        "4": {"class_type": "EmptySD3LatentImage" if job.get("model") in ("flux", "chroma") else "EmptyLatentImage", "inputs": {"width": job.get("width", 1024), "height": job.get("height", 1024), "batch_size": 1}},
    }
    if job.get("tile"):  # the tiling node is optional; plain images go straight from the loaders
        wf["9"] = {"class_type": "SeamlessTiling", "inputs": {"model": model_ref, "vae": vae_ref, "enabled": True}}
        model_ref, vae_ref = ["9", 0], ["9", 1]
    wf["5"] = {"class_type": "KSampler", "inputs": {
        "model": model_ref, "positive": ["2", 0], "negative": ["3", 0], "latent_image": ["4", 0],
        "seed": seed, "steps": job.get("steps", m["steps"]), "cfg": job.get("cfg", m["cfg"]),
        "sampler_name": m["sampler"], "scheduler": m["scheduler"], "denoise": 1.0}}
    wf["6"] = {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": vae_ref}}
    wf["7"] = {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "popfold/" + job["name"]}}
    return wf, seed


class VramWatch(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0
        self.stop = False

    def run(self):
        while not self.stop:
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                self.peak = max(self.peak, int(out.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.5)


def run(job):
    wf, seed = workflow(job)
    watch = VramWatch()
    watch.start()
    t0 = time.time()
    pid = api("/prompt", {"prompt": wf, "client_id": str(uuid.uuid4())})["prompt_id"]
    while True:
        h = api(f"/history/{pid}")
        if pid in h:
            entry = h[pid]
            break
        time.sleep(0.5)
    secs = time.time() - t0
    watch.stop = True
    status = entry.get("status", {})
    if status.get("status_str") != "success":
        raise RuntimeError(f"{job['name']} failed: {json.dumps(status)[:800]}")
    saved = []
    for node in entry["outputs"].values():
        for img in node.get("images", []):
            q = urllib.parse.urlencode({"filename": img["filename"], "subfolder": img["subfolder"], "type": img["type"]})
            data = urllib.request.urlopen(f"{SERVER}/view?{q}", timeout=60).read()
            dest = OUT / f"{job['name']}.png"
            dest.write_bytes(data)
            saved.append(dest)
    return {"name": job["name"], "model": job.get("model", "sdxl"), "seed": seed, "seconds": round(secs, 1),
            "peak_vram_mib": watch.peak, "file": str(saved[0]) if saved else ""}


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    only = set(sys.argv[2:])
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "results.csv"
    new = not log.exists()
    with log.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["name", "model", "seed", "seconds", "peak_vram_mib", "file"])
        if new:
            w.writeheader()
        for job in jobs:
            if only and job["name"] not in only:
                continue
            r = run(job)
            w.writerow(r)
            f.flush()
            print(f"{r['name']:<28} {r['model']:<5} {r['seconds']:>6}s  peak {r['peak_vram_mib']} MiB  -> {r['file']}", flush=True)


if __name__ == "__main__":
    main()
