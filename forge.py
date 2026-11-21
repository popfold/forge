"""A brief in, a checked prop out, made the same way every time, with local models only.

    python forge.py "a sprinkled donut" [--style toy] [--name NAME] [--seed 1] [--seeds 4] [--rounds 3]
                    [--pass 7] [--mesh trellis|hunyuan] [--shape tall|wide|square] [--tris N] [--fill] [--fresh]
    python forge.py --batch briefs.json [--fresh]       # every brief in a file, one after another

Needs ComfyUI (FLUX schnell, Hunyuan3D 2), Blender and Ollama with a vision model (README.md). Nothing leaves
this machine.

The script decides; the language model only answers questions. Each round runs fixed steps:

  plan     the model turns the brief into a subject, a triangle budget and things to check
  draw     FLUX draws --seeds pictures, seeds derived from --seed and the round
  measure  each picture's cut-out is measured: one object, not cut off, a sensible size, a white background,
           left-right symmetry. A picture that fails a measure is out, whatever the model thinks of it
  pick     the model scores the pictures; the best one that passed the measures wins (ties: symmetry, then seed)
  mesh     cut-out, Hunyuan3D 2, Blender low poly, review sheet (asset_pipeline.py's steps)
  check    the mesh's silhouette must match the picture (front IoU), then the model judges review.png
  next     on a fail, a rule picks the next round: a wrong drawing gets the model's rewritten subject and new
           seeds; a poor mesh gets the next-best picture and a 1.5x triangle budget

Repeatable: every step is keyed by a hash of what goes into it (prompt, seed, input files, the scripts doing
the work, the model's digest). A step whose key matches what is on disk is skipped, and model answers come
from llm.py's cache. So running the same command twice gives the same files, the second time in seconds, and a
run that stopped part way picks up where it stopped. --fresh redoes every step and asks the model again.

Everything lands in <out>/props/<name>/: r1/, r2/ ... (each round's pictures, mesh and sheet), the chosen
<name>.glb and review.png, and run.json (every input, measure, score and decision). Nothing is copied into a
site: a person looks at review.png and decides.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import asset_pipeline as ap
import comfy_3d
import comfy_run
import comfy_trellis
import config
import llm

HERE = Path(__file__).parent

# The fixed bars. A picture or mesh outside them fails, whatever the model says.
GATES = {
    "main_blob": 0.90,       # the largest cut-out piece holds at least this share of everything cut out
    "area": (0.04, 0.75),    # the object covers this share of the picture
    "white": 0.90,           # this share of the picture's border is near white (a soft floor shadow is fine)
    # the mesh's front silhouette overlaps the picture's this much. TRELLIS.2's is lower for a good model: it
    # rests objects as they would stand (a donut lies flat) and thin legs barely register on the coarse grid
    "front_iou": {"hunyuan": 0.55, "trellis": 0.45},
    "pick": 5,               # the model's lowest acceptable picture score (0-10)
}
FIGURE_TRIS = 3000           # a character, animal or person: at 1000-1500 a standing figure looks faceted

RULES = """You plan 3D props for a stylised miniature house. The pipeline: an image model draws the object from the
front on white, a mesh model turns that picture into a 3D model, and it is reduced to low poly.
Write the subject as a short plain description of ONE object (under 30 words): its shape, parts, colours and
materials. Describe only what it looks like: no rules, no "no ..." phrases, no words about style or originality.
Keep what makes the object itself (a donut keeps its hole, a mug its handle), made chunky.
A style wrapper adds the rendering words (view, lighting, background), so do not repeat them.
What works: chunky, solid, rounded parts. What fails: thin parts (wires, cords, slim legs, fine handles),
see-through glass, text, and anything that only reads from the side. Make thin parts thick on purpose.
Never name a brand, product line, company, franchise, character or real person, and never ask to copy an
existing design: describe shapes and colours instead. Our own designs only."""

PLAN = {"type": "object", "required": ["subject", "tris", "figure", "shape", "checks"], "properties": {
    "subject": {"type": "string", "maxLength": 220},
    "tris": {"type": "integer", "minimum": 600, "maximum": 6000},
    "figure": {"type": "boolean", "description": "true for a character, creature, animal or person figure; false for furniture and objects"},
    "shape": {"type": "string", "enum": ["tall", "wide", "square"],
              "description": "the object's outline seen from the front: tall (a floor lamp, a person), wide (a sofa, a sideboard) or square"},
    "checks": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 6,
               "description": "short things to look for in the finished model, from the brief"}}}

PICK = {"type": "object", "required": ["problems", "score"], "properties": {
    "problems": {"type": "string", "maxLength": 200, "description": "what in this picture does not fit the brief; empty if nothing"},
    "score": {"type": "integer", "minimum": 0, "maximum": 10}}}

LOOKER = """You check pictures for a pipeline that turns one picture of one object into a 3D model. Look carefully and
count parts before you say one is missing. Judge only what is in the picture against the brief."""

REWRITE = {"type": "object", "required": ["subject"], "properties": {"subject": {"type": "string", "maxLength": 220}}}

VERDICT = {"type": "object", "required": ["score", "passed", "problems", "redo"], "properties": {
    "score": {"type": "integer", "minimum": 0, "maximum": 10},
    "passed": {"type": "boolean"},
    "problems": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 6},
    "redo": {"type": "string", "enum": ["none", "picture", "mesh"],
             "description": "picture: the drawing itself is wrong (shape, colour, parts); mesh: the drawing was fine "
                            "but the 3D model is lumpy, holed or too coarse"}}}

TRELLIS_FACES = 20000    # TRELLIS.2's own output; Blender reduces it to the budget

FRESH = False
LIBRARY = config.OUT / "models"  # every model every round made, passed or not: <name>-r<round>-s<seed>.glb
_sess = None


# ---------- keys and steps ----------

def sha(data):
    return hashlib.sha256(data if isinstance(data, bytes) else json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def fhash(path):
    return sha(Path(path).read_bytes())


CODE = {n: fhash(HERE / n) for n in ("asset_pipeline.py", "comfy_run.py", "comfy_3d.py", "comfy_trellis.py", "blender_lowpoly.py",
                                     "blender_fit.py", "blender_render.py")}


def step(folder, name, key, outputs, fn):
    """Run fn() unless a step with the same key already wrote every output. Returns fn's result (kept as JSON)."""
    k = sha(key); stamp = folder / f".{name}.json"
    if not FRESH and stamp.exists() and all(Path(o).exists() for o in outputs):
        kept = json.loads(stamp.read_text(encoding="utf-8"))
        if kept["key"] == k:
            return kept["result"]
    folder.mkdir(parents=True, exist_ok=True)
    result = fn()
    stamp.write_text(json.dumps({"key": k, "result": result}, indent=2), encoding="utf-8")
    return result


def ask(*a, **kw):
    return llm.ask(*a, cache=not FRESH, **kw)


def banned(text):
    words = [w for w in config.LLM.get("banned", []) if w]
    return [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text, re.I)]


def positive(text):
    """Only what the object looks like. A small model writes rules into its subject ("no hole", "without text"): the
    picture model has no negative prompt and the picker then marks a donut down for its hole. Drop those clauses."""
    parts = [s.strip() for s in re.split(r"[,;]|\.\s", text)]
    keep = [s for s in parts if s and not re.match(r"(no|not|non|without|avoid|never|nothing)\b", s, re.I)]
    return ", ".join(keep).rstrip(" .")


def slug(brief):
    words = re.sub(r"[^a-z0-9]+", " ", brief.lower()).split()
    while words and words[0] in ("a", "an", "the", "one", "some"):
        words = words[1:]
    return "-".join(words[:5]) or "prop"


# ---------- measures (no model involved) ----------

def measure(path):
    """The cut-out of one picture, in numbers, and which gates it fails."""
    global _sess
    from rembg import new_session, remove
    _sess = _sess or new_session("u2net")
    rgb = Image.open(path).convert("RGB")
    cut = np.array(remove(rgb, session=_sess))[..., 3] > 128
    H, W = cut.shape
    lab, n = ndimage.label(cut)
    if not n:
        return {"fails": ["nothing cut out"]}
    sizes = ndimage.sum(cut, lab, range(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.where(keep)
    crop = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # the border, not the whole background: furniture drawn on white casts a soft floor shadow the cut-out leaves
    border = np.zeros_like(cut); b = max(4, H // 12)
    border[:b] = border[-b:] = True; border[:, :b] = border[:, -b:] = True
    bg = np.array(rgb)[border & ~ndimage.binary_dilation(cut, iterations=6)]
    m = {
        "main_blob": round(float(sizes.max() / sizes.sum()), 3),
        "area": round(float(keep.sum() / (H * W)), 3),
        "edge": bool(ys.min() == 0 or xs.min() == 0 or ys.max() == H - 1 or xs.max() == W - 1),
        "white": round(float((bg.min(axis=1) > 235).mean()) if len(bg) else 0.0, 3),
        "symmetry": round(float((crop & crop[:, ::-1]).sum() / max(1, (crop | crop[:, ::-1]).sum())), 3),
    }
    lo, hi = GATES["area"]
    m["fails"] = [f for f, bad in (
        ("more than one object", m["main_blob"] < GATES["main_blob"]),
        ("cut off at the edge", m["edge"]),
        ("too small or too big", not lo <= m["area"] <= hi),
        ("background not white", m["white"] < GATES["white"])) if bad]
    return m


def mesh_numbers(lines):
    text = "\n".join(lines)
    num = lambda pat, cast=float: cast(re.search(pat, text).group(1)) if re.search(pat, text) else None  # noqa: E731
    return {"islands": num(r"islands=(\d+)", int), "kept": num(r"kept=(\d+)", int),
            "front_iou": num(r"(?:front: right=|FIT .* front=)\S+ iou=([\d.]+)"), "tris": num(r"(?:LOWPOLY|FIT) .* tris=(\d+)", int),
            "bytes": num(r"bytes=(\d+)", int)}


# ---------- the model's questions ----------

COLOURS = {"red", "orange", "yellow", "mustard", "gold", "green", "olive", "teal", "turquoise", "blue", "navy", "purple",
           "violet", "lilac", "pink", "magenta", "brown", "walnut", "oak", "tan", "beige", "cream", "ivory", "white", "grey",
           "gray", "black", "silver", "brass", "copper", "mint", "sage", "coral", "peach", "burgundy", "maroon", "lavender"}


def colours(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w in COLOURS}


def plan(brief, style, seed):
    user = f"Brief: {brief}\nStyle: {style.get('about', '')}\nPlan the job."
    # a subject that drops most of the brief is no plan: asked for a walnut, cream and brass floor lamp, the model
    # once answered just "floor-standing lamp". Ask again; in the end draw from the brief itself
    short = max(6, len(brief.split()) // 3)
    for attempt in range(3):
        p = ask(RULES, user, PLAN, seed=seed + attempt)
        p["subject"] = positive(p["subject"])
        p["checks"] = [c for c in p["checks"] if positive(c) == c.strip().rstrip(".")]
        hit = banned(p["subject"])
        # colours too: a dinosaur brief's "teal green body, cream belly" once became a subject with neither, and FLUX
        # drew grey and yellow dinosaurs
        lost = sorted(colours(brief) - colours(p["subject"]))
        if p["subject"] and not hit and len(p["subject"].split()) >= short and not lost:
            break
        user += (f"\nYour last subject used a word that is not allowed ({', '.join(hit)}). Describe it without it." if hit
                 else f"\nYour last subject left out the brief's colours ({', '.join(lost)}). Keep every colour and part."
                 if lost else "\nYour last subject left out most of the brief. Keep its parts, colours and materials.")
    else:
        if hit:
            raise SystemExit(f"the planner kept using banned words: {hit}")
        p["subject"] = positive(brief)
    p["tris"] = max(600, min(6000, int(p["tris"])))
    if p.get("figure"):
        p["tris"] = max(p["tris"], FIGURE_TRIS)
    return p


def pick(brief, subject, files, seed, style):
    """Score each picture on its own. Shown four at once, an 8B model sees each one small and miscounts: it said
    four sofas with three back cushions each had two. Problems come first in the answer so the score follows them."""
    user = (f"Brief: {brief}\nStyle: {style.get('about', '')}\nThis picture should show exactly one such object, whole, "
            "seen from the front or a little to one side, on white, with parts solid enough for a 3D model to rebuild. "
            "Say what does not fit the brief, then score it 0-10: 8-10 fits the brief well, 5-7 is usable with small "
            "differences, under 5 is the wrong object, the wrong colours, cut off or more than one object.")
    out = []
    for i, f in enumerate(files):
        a = ask(LOOKER, user, PICK, images=[f], seed=seed)
        out.append({"image": i + 1, "score": int(a["score"]), "problems": a["problems"]})
    return out


def rewrite(brief, subject, problems, seed):
    user = (f"Brief: {brief}\nSubject drawn: {subject}\nThe pictures or the model made from them had these problems: "
            f"{'; '.join(problems) or 'none usable'}\nRewrite the subject so the next pictures avoid them.")
    for attempt in range(3):
        s = positive(ask(RULES, user, REWRITE, seed=seed + attempt)["subject"])
        if s and not banned(s):
            return s
    return subject


def judge(brief, job, sheet, seed, style):
    user = (f"Brief: {brief}\nSubject drawn: {job['subject']}\nTriangle budget: {job['tris']}\n"
            f"Checks: {'; '.join(job.get('checks', [])) or 'none'}\n"
            "This sheet shows the finished low-poly 3D model from four sides. The back is always a guess and a little "
            "blurry, so do not fail it for that. Fail it for: wrong shape or colours for the brief, missing or merged "
            "parts, lumps, holes, spikes, a melted look, or parts too thin to read. It is low poly and will be seen small "
            "on screen, so small details (knobs, seams, grain, exact shades) do not matter. Pass it when it would look "
            f"right as a small prop in a miniature house in this style: {style.get('about', '')} "
            "Score it 0-10. Keep each problem to one short sentence.")
    return ask(RULES, user, VERDICT, images=[sheet], seed=seed)


# ---------- the GPU steps ----------

# The picture's own shape: on a square canvas FLUX drew "a tall floor lamp, as tall as a person" as a table lamp
# eight times in a row. A tall canvas lets a tall thing be tall.
CANVAS = {"tall": (768, 1344), "wide": (1344, 768), "square": (1024, 1024)}


def draw(folder, prompt, seed, shape="square"):
    out = folder / f"s{seed}.png"
    w, h = CANVAS[shape]

    def go():
        comfy_run.OUT = folder
        r = comfy_run.run({"name": f"s{seed}", "model": "flux", "seed": seed, "prompt": prompt, "width": w, "height": h})
        return {"seconds": r["seconds"]}
    step(folder, f"draw-s{seed}", {"prompt": prompt, "seed": seed, "size": [w, h], "ckpt": comfy_run.MODELS["flux"],
                                   "code": CODE["comfy_run.py"]}, [out], go)
    return out


def make_mesh(rdir, source, tris, fill, engine="trellis"):
    shutil.copy(source, rdir / "source.png")
    dense = ".hunyuan.json" if engine == "hunyuan" else ".trellis.json"
    # an earlier round with the same picture already has its cut-out and mesh: take them, with their keys
    for prev in sorted(rdir.parent.glob("r*")):
        if prev != rdir and (prev / dense).exists() and (prev / "source.png").exists()                 and fhash(prev / "source.png") == fhash(source) and not (rdir / dense).exists():
            for f in ("prepared.png", "mask.png", "texture.png", "raw.glb", ".prepare.json", dense):
                if (prev / f).exists():
                    shutil.copy(prev / f, rdir / f)
            break
    prep = step(rdir, "prepare", {"src": fhash(source), "fill": fill, "code": CODE["asset_pipeline.py"]},
                [rdir / "prepared.png", rdir / "mask.png", rdir / "texture.png"],
                lambda: ap.prepare(rdir / "source.png", rdir / "prepared.png", rdir / "mask.png", fill=fill) or {})
    glb = rdir / "prop.glb"
    if engine == "trellis":
        # TRELLIS.2 makes the low-poly, textured model itself; Blender only turns and sizes it
        def trellis():
            comfy_trellis.OUT = rdir
            # ComfyUI's decimation is quick and crude (it crumpled a chair at 1000 faces): stop at a dense
            # mesh, bake the colours onto it, and let Blender reduce it to the budget in the fit step
            r = comfy_trellis.run(rdir / "prepared.png", faces=TRELLIS_FACES, mask=rdir / "mask.png")
            Path(r["file"]).replace(rdir / "raw.glb")
            return {"seconds": r["seconds"], "peak_vram_mib": r["peak_vram_mib"], "fine": r["fine"]}
        step(rdir, "trellis", {"prepared": fhash(rdir / "prepared.png"), "mask": fhash(rdir / "mask.png"),
                               "faces": TRELLIS_FACES, "files": comfy_trellis.FILES,
                               "code": CODE["comfy_trellis.py"]}, [rdir / "raw.glb"], trellis)

        def fit():
            lines = ap.blender("blender_fit.py", rdir / "raw.glb", rdir / "mask.png", glb, "--tris", tris)
            ap.blender("blender_render.py", glb, rdir / "review.png", "--views", 4)
            return mesh_numbers(lines)
        return step(rdir, "fit", {"raw": fhash(rdir / "raw.glb"), "mask": fhash(rdir / "mask.png"), "tris": tris,
                                  "code": [CODE["blender_fit.py"], CODE["blender_render.py"], CODE["asset_pipeline.py"]]},
                    [glb, rdir / "review.png"], fit)

    def hunyuan():
        comfy_3d.OUT = rdir
        r = comfy_3d.run(rdir / "prepared.png")
        Path(r["file"]).replace(rdir / "raw.glb")
        return {"seconds": r["seconds"]}
    step(rdir, "hunyuan", {"prepared": fhash(rdir / "prepared.png"), "ckpt": comfy_3d.CKPT, "code": CODE["comfy_3d.py"]},
         [rdir / "raw.glb"], hunyuan)

    def lowpoly():
        lines = ap.blender("blender_lowpoly.py", rdir / "raw.glb", rdir / "texture.png", rdir / "mask.png", glb, "--tris", tris)
        ap.blender("blender_render.py", glb, rdir / "review.png", "--views", 4)
        return mesh_numbers(lines)
    return step(rdir, "lowpoly", {"raw": fhash(rdir / "raw.glb"), "texture": fhash(rdir / "texture.png"), "tris": tris,
                                  "code": [CODE["blender_lowpoly.py"], CODE["blender_render.py"]], "prep": prep},
                [glb, rdir / "review.png"], lowpoly)


# ---------- progress pictures ----------

def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow before 10.1 has one size only
        return ImageFont.load_default()


def progress(d, label, tiles, title):
    """One picture per stage in <name>/progress/, numbered in order: tiles are (image path, caption, colour)."""
    folder = d / "progress"; folder.mkdir(exist_ok=True)
    n = len(list(folder.glob("*.jpg"))) + 1
    h = 360; pad = 12; cap = 48
    ims = []
    for path, caption, colour in tiles:
        im = Image.open(path).convert("RGB"); im = im.resize((max(1, round(im.width * h / im.height)), h))
        ims.append((im, caption, colour))
    W = sum(im.width for im, _, _ in ims) + pad * (len(ims) + 1); H = 56 + h + cap + pad
    sheet = Image.new("RGB", (W, H), "white"); draw = ImageDraw.Draw(sheet)
    draw.text((pad, 14), title, fill="black", font=_font(26))
    x = pad
    for im, caption, colour in ims:
        sheet.paste(im, (x, 56))
        draw.rectangle([x - 3, 53, x + im.width + 2, 56 + h + 2], outline=colour, width=5)
        f = _font(20)
        while len(caption) > 4 and draw.textlength(caption, font=f) > im.width:  # keep each caption under its picture
            caption = caption[:-2].rstrip() + "…"
        draw.text((x, 56 + h + 10), caption, fill=colour if colour != "grey" else "black", font=f)
        x += im.width + pad
    out = folder / f"{n:02d}-{label}.jpg"
    sheet.save(out, quality=88)
    print(f"progress {out}", flush=True)
    return out


# ---------- the run ----------

def versions():
    v = {"llm": config.LLM.get("model"), "llm_digest": llm.digest()[:16], "code": CODE}
    try:
        with urllib.request.urlopen(config.COMFY_URL + "/system_stats", timeout=10) as r:
            v["comfyui"] = json.loads(r.read())["system"]["comfyui_version"]
    except Exception as e:
        raise SystemExit(f"ComfyUI is not answering at {config.COMFY_URL}: {e}")
    return v


def main():
    global FRESH
    p = argparse.ArgumentParser(description="A brief in, a checked prop out, with local models only.")
    p.add_argument("brief", nargs="+")
    p.add_argument("--style", default=ap.DEFAULT_STYLE)
    p.add_argument("--name")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--pass", dest="pass_mark", type=int, default=7)
    p.add_argument("--mesh", choices=["trellis", "hunyuan"], default="trellis",
                   help="trellis: shape and colours on every side (slower); hunyuan: shape only, the picture projected on")
    p.add_argument("--tris", type=int, help="the triangle budget, instead of the planner's")
    p.add_argument("--shape", choices=list(CANVAS), help="the picture's shape, instead of the planner's")
    p.add_argument("--fill", action="store_true", help="keep pale or see-through panels the cut-out punches out")
    p.add_argument("--fresh", action="store_true")
    a = p.parse_args()
    FRESH = a.fresh
    brief = " ".join(a.brief)
    if banned(brief):
        raise SystemExit(f"the brief uses a word that is not allowed: {banned(brief)}")
    style = ap.STYLES[a.style]
    name = a.name or slug(brief)
    d = ap.OUT / name; d.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(d / "progress", ignore_errors=True)  # this run's pictures only
    log = {"brief": brief, "name": name, "mesh": a.mesh, "style": a.style, "seed": a.seed, "seeds": a.seeds, "rounds": a.rounds,
           "pass": a.pass_mark, "gates": GATES, "versions": versions(), "round": []}
    save = lambda: (d / "run.json").write_text(json.dumps(log, indent=2), encoding="utf-8")  # noqa: E731
    say = lambda s: print(s, flush=True)  # noqa: E731

    job = plan(brief, style, a.seed) | {"fill": a.fill}
    if a.tris:
        job["tris"] = max(600, min(6000, a.tris))
    job["shape"] = a.shape or job.get("shape") or "square"
    log["plan"] = dict(job); save()
    say(f"plan     {name}: {job['subject']}  (tris {job['tris']}, {job['shape']} picture, fill {job['fill']})")
    subject, tris, pool, best = job["subject"], job["tris"], [], None
    for rnd in range(1, a.rounds + 1):
        rdir = d / f"r{rnd}"; rdir.mkdir(exist_ok=True)
        entry = {"round": rnd, "subject": subject, "tris": tris}
        log["round"].append(entry)
        if not pool:  # draw, measure, pick
            prompt = style["prompt"].format(subject=subject)
            seeds = [a.seed * 1000 + rnd * 10 + k for k in range(a.seeds)]
            files = [draw(rdir / "candidates", prompt, s, job["shape"]) for s in seeds]
            meas = [step(rdir / "candidates", f"measure-s{s}", {"img": fhash(f), "gates": GATES}, [f], lambda f=f: measure(f))
                    for s, f in zip(seeds, files)]
            scores = pick(brief, subject, files, a.seed + rnd, style)
            cands = [{"seed": s, "file": str(f), "measure": m, "score": sc["score"], "problems": sc["problems"]}
                     for s, f, m, sc in zip(seeds, files, meas, scores)]
            entry["candidates"] = cands
            for c in cands:
                say(f"picture  r{rnd} s{c['seed']}: score {c['score']}, symmetry {c['measure'].get('symmetry')}"
                    + (f" -- out: {', '.join(c['measure']['fails'])}" if c["measure"]["fails"] else ""))
            pool = sorted((c for c in cands if not c["measure"]["fails"] and c["score"] >= GATES["pick"]),
                          key=lambda c: (-c["score"], -c["measure"]["symmetry"], c["seed"]))
            top = pool[0]["seed"] if pool else None
            progress(d, f"r{rnd}-pictures", [(c["file"], f"s{c['seed']}  score {c['score']}"
                                              + (f"  out: {', '.join(c['measure']['fails'])}" if c["measure"]["fails"] else ""),
                                              "green" if c["seed"] == top else "red" if c["measure"]["fails"] else "grey")
                                             for c in cands],
                     f"Round {rnd}: pictures. " + (f"Chosen: s{top}" if top else "None usable: rewriting the subject"))
            if not pool:
                entry["decision"] = "no usable picture: rewrite the subject"; save()
                say(f"next     r{rnd}: no usable picture, rewriting the subject")
                subject = rewrite(brief, subject, [c["problems"] for c in cands] +
                                  [", ".join(c["measure"]["fails"]) for c in cands if c["measure"]["fails"]], a.seed + rnd)
                continue
        chosen = pool.pop(0)
        entry["chosen"] = chosen["seed"]
        mesh = make_mesh(rdir, Path(chosen["file"]), tris, job["fill"], a.mesh)
        entry["mesh"] = mesh
        LIBRARY.mkdir(parents=True, exist_ok=True)
        keep = f"{name}-r{rnd}-s{chosen['seed']}"
        shutil.copy(rdir / "prop.glb", LIBRARY / f"{keep}.glb")
        if (rdir / "raw.glb").exists():
            shutil.copy(rdir / "raw.glb", LIBRARY / f"{keep}-dense.glb")
        entry["library"] = f"{keep}.glb"
        bar = GATES["front_iou"][a.mesh]
        fails = [] if (mesh.get("front_iou") or 0) >= bar else [f"front silhouette {mesh.get('front_iou')} under {bar}"]
        verdict = judge(brief, job | {"subject": subject, "tris": tris}, rdir / "review.png", a.seed + rnd, style)
        passed = not fails and verdict["passed"] and verdict["score"] >= a.pass_mark
        entry |= {"gate_fails": fails, "verdict": verdict, "passed": passed}
        say(f"judge    r{rnd} s{chosen['seed']}: {verdict['score']}/10, front {mesh.get('front_iou')}, "
            f"{mesh.get('tris')} tris -- {'PASS' if passed else 'FAIL'}"
            + (f": {'; '.join(fails + verdict['problems'])}" if (fails or verdict["problems"]) and not passed else ""))
        progress(d, f"r{rnd}-model", [(rdir / "prepared.png", f"cut-out of s{chosen['seed']}", "grey"),
                                      (rdir / "review.png", f"{mesh.get('tris')} triangles, front match {mesh.get('front_iou')}",
                                       "green" if passed else "red")],
                 f"Round {rnd}: model. Judge {verdict['score']}/10, {'PASS' if passed else 'FAIL'}"
                 + ("" if passed else f": {'; '.join(fails + verdict['problems'])[:110]}"))
        rank = (passed, verdict["score"], mesh.get("front_iou") or 0)
        if best is None or rank > best[0]:
            best = (rank, rnd)
        if passed:
            save(); break
        if rnd == a.rounds:
            save(); break
        # the rule for the next round
        if verdict["redo"] == "picture" and not fails:
            entry["decision"] = "wrong drawing: rewrite the subject, new seeds"
            subject = rewrite(brief, subject, verdict["problems"], a.seed + rnd); pool = []
        else:
            tris = min(6000, int(tris * 1.5))
            entry["decision"] = f"poor mesh: {'next-best' if pool else 'same'} picture, {tris} triangles"
            if not pool:
                pool = [chosen]
        say(f"next     r{rnd}: {entry['decision']}")
        save()

    if best is None:
        log["result"] = {"passed": False, "why": "no round got as far as a mesh"}; save()
        raise SystemExit(f"done     NOT passed: no usable picture in {a.rounds} rounds. See {d / 'run.json'}")
    rdir = d / f"r{best[1]}"
    shutil.copy(rdir / "prop.glb", d / f"{name}.glb"); shutil.copy(rdir / "review.png", d / "review.png")
    log["result"] = {"passed": best[0][0], "round": best[1], "glb": str(d / f"{name}.glb"),
                     "review": str(d / "review.png"), "sha": fhash(d / f"{name}.glb")}
    save()
    say(f"done     {'passed' if best[0][0] else 'NOT passed (best round kept)'}: {d / f'{name}.glb'} (round {best[1]})\n"
        f"         look at {d / 'review.png'} before using it")


def batch(path, extra):
    """Run every brief in a JSON file, one at a time (the GPU holds one job). Each entry is
    {"name", "brief", "style"?, "shape"?, "tris"?, "seed"?, "seeds"?, "rounds"?, "pass"?, "mesh"?, "fill"?}.
    The same file always makes the same props: every step is keyed and kept, so a second run takes seconds."""
    import subprocess
    jobs = json.loads(Path(path).read_text(encoding="utf-8"))
    done = []
    for j in jobs:
        args = [sys.executable, str(Path(__file__).resolve()), "--name", j["name"], *extra]
        for k in ("style", "shape", "tris", "seed", "seeds", "rounds", "pass", "mesh"):
            if k in j:
                args += [f"--{k}", str(j[k])]
        if j.get("fill"):
            args.append("--fill")
        print(f"=== START {j['name']}", flush=True)
        r = subprocess.run(args + [j["brief"]])
        done.append((j["name"], r.returncode))
        print(f"=== END {j['name']} ({'ok' if r.returncode == 0 else 'not passed'})", flush=True)
    print("=== BATCH DONE  " + ", ".join(f"{n}: {'ok' if c == 0 else 'not passed'}" for n, c in done), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--batch":
        batch(sys.argv[2], sys.argv[3:])  # extra flags (say --fresh) apply to every entry
    else:
        main()
