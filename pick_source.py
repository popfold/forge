"""Draw several candidate source images and keep the most straight-on one (left-right symmetric cut-out).

    python pick_source.py assets.json --out=DIR --style=NAME --seeds=4 name [name ...]
"""
import json, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image
import asset_pipeline as ap
import comfy_run

args = [a for a in sys.argv[2:] if not a.startswith("--")]
opts = dict(a[2:].split("=", 1) for a in sys.argv[2:] if a.startswith("--") and "=" in a)
out = Path(opts["out"]); style = opts.get("style", "toy"); n = int(opts.get("seeds", 4))
from rembg import new_session, remove
sess = new_session("u2net")
for a in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8")):
    if a["name"] not in args:
        continue
    d = out / a["name"] / "candidates"; d.mkdir(parents=True, exist_ok=True)
    comfy_run.OUT = d
    best = None
    for k in range(n):
        seed = a.get("seed", 1) * 10 + k
        comfy_run.run({"name": f"s{seed}", "model": "flux", "seed": seed,
                       "prompt": ap.STYLES[a.get("style", style)]["prompt"].format(subject=a["prompt"])})
        m = np.array(remove(Image.open(d / f"s{seed}.png").convert("RGB"), session=sess))[..., 3] > 128
        ys, xs = np.where(m); m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        sym = (m & m[:, ::-1]).sum() / max(1, (m | m[:, ::-1]).sum())
        print(f"{a['name']} seed {seed} symmetry {sym:.3f}", flush=True)
        if best is None or sym > best[0]:
            best = (sym, seed)
    shutil.copy(d / f"s{best[1]}.png", out / a["name"] / "source.png")
    print(f"{a['name']} picked seed {best[1]} (symmetry {best[0]:.3f})", flush=True)
