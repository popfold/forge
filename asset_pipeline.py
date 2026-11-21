"""Prompt -> low-poly textured .glb, end to end.

    python asset_pipeline.py assets.json [name ...]      # full pipeline
    python asset_pipeline.py assets.json --from-image    # reuse existing source images

For each asset {name, prompt, tris?, seed?, flat?}:
  1. FLUX draws a straight-on, shadow-free product image          (ComfyUI)
  2. background removed, object centred on white with a margin     (rembg)
  3. Hunyuan3D 2 turns it into a dense mesh                        (ComfyUI)
  4. Blender cleans it, reduces it to low poly and projects colour (blender_lowpoly.py)
  5. Blender renders a review sheet                                (blender_render.py)
Outputs land in <out>/props/<name>/ (see config.py for <out>, the ComfyUI address and the Blender path).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import comfy_3d
import comfy_run
import config

HERE = Path(__file__).parent
OUT = config.OUT / "props"
BLENDER = config.BLENDER
# A style is a file in styles/: the prompt wrapper around each asset's subject, and a default triangle budget.
STYLES = config.styles()
DEFAULT_STYLE = "toy"
PALETTE = False  # --palette: flat part colours instead of the projected photo
_session = None


def prepare(src: Path, dst_img: Path, dst_mask: Path, size=1024, border=0.12, fill=False):
    """Cut out the object, centre it on white with a margin, and bleed its edge colours outward."""
    global _session
    from rembg import new_session, remove
    _session = _session or new_session("u2net")
    rgba = remove(Image.open(src).convert("RGB"), session=_session)
    a = np.array(rgba)[..., 3] > 128
    lab, n = ndimage.label(a)
    if n > 1:  # keep the largest blob only
        a = lab == (np.argmax(ndimage.sum(a, lab, range(1, n + 1))) + 1)
    if fill:  # the cut-out sometimes punches out pale glass or panels; "fill": true keeps them
        a = ndimage.binary_fill_holes(a)
    ys, xs = np.where(a)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    rgb = np.array(Image.open(src).convert("RGB"))[y0:y1, x0:x1]; m = a[y0:y1, x0:x1]
    s = int(size * (1 - 2 * border)); k = s / max(m.shape)
    nw, nh = max(1, round(m.shape[1] * k)), max(1, round(m.shape[0] * k))
    rgb = np.array(Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS))
    m = np.array(Image.fromarray((m * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)) > 128
    canvas = np.full((size, size, 3), 255, np.uint8); mask = np.zeros((size, size), bool)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw][m] = rgb[m]; mask[oy:oy + nh, ox:ox + nw] = m
    Image.fromarray(canvas).save(dst_img)                       # what Hunyuan3D sees
    Image.fromarray((mask * 255).astype(np.uint8)).save(dst_mask)
    # texture atlas: left half = the image (colours bled past the silhouette so UV edges never
    # sample white); right half = a heavily blurred copy, used for faces the image can't see
    inner = ndimage.binary_erosion(mask, iterations=3)
    idx = ndimage.distance_transform_edt(~inner, return_distances=False, return_indices=True)
    sharp = canvas[idx[0], idx[1]]
    w = inner.astype(np.float32)
    sig = size * 0.085
    num = np.stack([ndimage.gaussian_filter(canvas[..., c] * w, sig) for c in range(3)], -1)
    soft = (num / np.maximum(ndimage.gaussian_filter(w, sig), 1e-4)[..., None]).clip(0, 255).astype(np.uint8)
    soft = soft[idx[0], idx[1]]
    atlas = np.concatenate([sharp, soft], 1)
    Image.fromarray(atlas).resize((1536, 768), Image.LANCZOS).save(dst_img.with_name("texture.png"))


def blender(script, *args):
    r = subprocess.run([BLENDER, "-b", "-P", str(HERE / script), "--", *map(str, args)], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith(("LOWPOLY", "FIT", "RENDERED", "front:", "islands", "  side", "ANIM"))]
    if r.returncode or not lines:
        raise RuntimeError(r.stdout[-1500:] + r.stderr[-1500:])
    return lines


def step_image(asset):
    d = OUT / asset["name"]; d.mkdir(parents=True, exist_ok=True)
    comfy_run.OUT = d
    return comfy_run.run({"name": "source", "model": "flux", "seed": asset.get("seed", 1),
                          "prompt": asset.get("raw_prompt") or STYLES[asset.get("style", DEFAULT_STYLE)]["prompt"].format(subject=asset["prompt"])})["seconds"]


def step_mesh(asset):
    d = OUT / asset["name"]
    prepare(d / "source.png", d / "prepared.png", d / "mask.png", fill=asset.get("fill", False))
    comfy_3d.OUT = d
    r = comfy_3d.run(d / "prepared.png")
    Path(r["file"]).replace(d / "raw.glb")
    return r["seconds"]


def step_blender(asset):
    name = asset["name"]; d = OUT / name; out = d / f"{name}.glb"
    args = [d / "raw.glb", d / "texture.png", d / "mask.png", out, "--tris", asset.get("tris", STYLES[asset.get("style", DEFAULT_STYLE)].get("tris", 2500))] + (["--flat"] if asset.get("flat") else []) + (["--depth", asset["depth"]] if asset.get("depth") else []) + (["--palette", "--colours", asset.get("colours", 7)] if asset.get("palette", PALETTE) else [])
    lines = blender("blender_lowpoly.py", *args)
    blender("blender_render.py", out, d / "review.png", "--views", 4)
    return lines[-1]


if __name__ == "__main__":
    assets = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    flags = {a for a in sys.argv[2:] if a.startswith("--")}
    for f in flags:  # --out=DIR picks the output folder, --style=NAME the default prompt style
        if f.startswith("--out="):
            OUT = Path(f[6:])
        if f == "--palette":
            PALETTE = True
        if f.startswith("--style="):
            DEFAULT_STYLE = f[8:]
    only = {a for a in sys.argv[2:] if not a.startswith("--")}
    assets = [a for a in assets if not only or a["name"] in only]
    logs = {a["name"]: {"name": a["name"]} for a in assets}
    # One phase at a time for the whole batch, so the GPU loads each model once instead of
    # swapping FLUX and Hunyuan3D in and out for every asset.
    phases = [("image_s", step_image), ("mesh_s", step_mesh), ("blender", step_blender)]
    if "--from-image" in flags:
        phases = phases[1:]
    if "--from-mesh" in flags:
        phases = phases[2:]
    failed = set()
    for key, fn in phases:
        for a in assets:
            if a["name"] in failed:
                continue
            try:
                logs[a["name"]][key] = fn(a)
                print(f"{key:<8} {a['name']:<22} {logs[a['name']][key]}", flush=True)
            except Exception as e:  # keep the batch going; report at the end
                failed.add(a["name"]); print(f"{key:<8} {a['name']:<22} FAILED: {str(e)[-500:]}", flush=True)
    for a in assets:
        (OUT / a["name"] / "log.json").write_text(json.dumps(logs[a["name"]], indent=2))
    print(f"done: {len(assets) - len(failed)} ok, {len(failed)} failed {sorted(failed)}")
