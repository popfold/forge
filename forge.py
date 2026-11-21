"""A brief in, a checked prop out, made the same way every time, with local models only.

    python forge.py "a sprinkled donut" [--style toy] [--name NAME] [--seed 1] [--seeds 4] [--rounds 3]
                    [--pass 7] [--fresh]

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
    "front_iou": {"hunyuan": 0.55, "trellis": 0.35},
    "pick": 5,               # the model's lowest acceptable picture score (0-10)
}

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

PLAN = {"type": "object", "required": ["subject", "tris", "checks"], "properties": {
    "subject": {"type": "string", "maxLength": 220},
    "tris": {"type": "integer", "minimum": 600, "maximum": 6000},
    "checks": {"type": "array", "items": {"type": "string", "maxLength": 120}, "maxItems": 6,
               "description": "short things to look for in the finished model, from the brief"}}}

PICK = {"type": "object", "required": ["scores"], "properties": {
    "scores": {"type": "array", "items": {"type": "object", "required": ["image", "score", "problems"], "properties": {
        "image": {"type": "integer"}, "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "problems": {"type": "string", "maxLength": 200}}}}}}

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

def plan(brief, style, seed):
    user = f"Brief: {brief}\nStyle: {style.get('about', '')}\nPlan the job."
    for attempt in range(3):
        p = ask(RULES, user, PLAN, seed=seed + attempt)
        p["subject"] = positive(p["subject"])
        p["checks"] = [c for c in p["checks"] if positive(c) == c.strip().rstrip(".")]
        hit = banned(p["subject"])
        if p["subject"] and not hit:
            p["tris"] = max(600, min(6000, int(p["tris"])))
            return p
        user += f"\nYour last subject used a word that is not allowed ({', '.join(hit)}). Describe it without it."
    raise SystemExit(f"the planner kept using banned words: {hit}")


def pick(brief, subject, files, seed):
    user = (f"Brief: {brief}\nSubject drawn: {subject}\nThese {len(files)} pictures are candidates, numbered 1 to "
            f"{len(files)} in order. Each must show exactly one object that fits the brief, whole, straight from the "
            "front, on white, with chunky parts a mesh model can rebuild. Compare them against each other and the brief "
            "(colours, which part is which colour, shape). Give every picture a different score; no ties.")
    # the schema asks for exactly one score per picture, numbered 1..n: left open, the model once answered with an
    # empty list and every picture counted as unusable. Still short? Ask again with the next seed.
    n = len(files)
    schema = json.loads(json.dumps(PICK)); arr = schema["properties"]["scores"]
    arr["minItems"] = arr["maxItems"] = n
    arr["items"]["properties"]["image"] |= {"minimum": 1, "maximum": n}
    got = {}
    for attempt in range(3):
        got = {s["image"]: s for s in ask(RULES, user, schema, images=files, seed=seed + 100 * attempt)["scores"]}
        if all(i + 1 in got for i in range(n)):
            break
    return [got.get(i + 1, {"image": i + 1, "score": 0, "problems": "not scored"}) for i in range(n)]


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

def draw(folder, prompt, seed):
    out = folder / f"s{seed}.png"

    def go():
        comfy_run.OUT = folder
        r = comfy_run.run({"name": f"s{seed}", "model": "flux", "seed": seed, "prompt": prompt})
        return {"seconds": r["seconds"]}
    step(folder, f"draw-s{seed}", {"prompt": prompt, "seed": seed, "ckpt": comfy_run.MODELS["flux"], "code": CODE["comfy_run.py"]},
         [out], go)
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
            r = comfy_trellis.run(rdir / "prepared.png", faces=TRELLIS_FACES)
            Path(r["file"]).replace(rdir / "raw.glb")
            return {"seconds": r["seconds"], "peak_vram_mib": r["peak_vram_mib"]}
        step(rdir, "trellis", {"prepared": fhash(rdir / "prepared.png"), "faces": TRELLIS_FACES, "files": comfy_trellis.FILES,
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
    log["plan"] = dict(job); save()
    say(f"plan     {name}: {job['subject']}  (tris {job['tris']}, fill {job['fill']})")
    subject, tris, pool, best = job["subject"], job["tris"], [], None
    for rnd in range(1, a.rounds + 1):
        rdir = d / f"r{rnd}"; rdir.mkdir(exist_ok=True)
        entry = {"round": rnd, "subject": subject, "tris": tris}
        log["round"].append(entry)
        if not pool:  # draw, measure, pick
            prompt = style["prompt"].format(subject=subject)
            seeds = [a.seed * 1000 + rnd * 10 + k for k in range(a.seeds)]
            files = [draw(rdir / "candidates", prompt, s) for s in seeds]
            meas = [step(rdir / "candidates", f"measure-s{s}", {"img": fhash(f), "gates": GATES}, [f], lambda f=f: measure(f))
                    for s, f in zip(seeds, files)]
            scores = pick(brief, subject, files, a.seed + rnd)
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


if __name__ == "__main__":
    main()
