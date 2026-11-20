"""Wall art for picture frames: python art_run.py art-jobs.json -> <out>/art/<theme>/raw/<name>.png (see config.py).

Each job: {"theme", "name", "prompt", "seed", "frame": "main" | "square" | "small"} or "width" and "height" of its own.
"""
import json, sys
from pathlib import Path
import comfy_run
import config

# three frame shapes: a tall main picture, a square and a small portrait
SHAPES = {"main": (896, 1088), "square": (1024, 1024), "small": (832, 1152)}
TAIL = ", flat 2D artwork filling the whole image edge to edge, no frame, no border, no wall, no text, no signature"
only = set(sys.argv[2:])
for j in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    if only and j["name"] not in only:
        continue
    out = config.OUT / "art" / j["theme"] / "raw"; out.mkdir(parents=True, exist_ok=True)
    comfy_run.OUT = out
    w, h = (j["width"], j["height"]) if "width" in j else SHAPES[j["frame"]]
    r = comfy_run.run({"name": j["name"], "model": "flux", "seed": j["seed"], "width": w, "height": h, "prompt": j["prompt"] + TAIL})
    print(j["theme"], j["name"], r["seconds"], flush=True)
