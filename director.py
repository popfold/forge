"""A brief goes in, a checked prop comes out: a local language model plans the job, the pipeline makes it,
and the model judges the pictures and the finished review sheet, retrying until it passes.

    python director.py "a mint bedside lamp shaped like a mushroom" [--style toy] [--name NAME]
                       [--rounds 3] [--seeds 3] [--seed 1] [--pass 8]
    python director.py --judge DIR "the brief"       # only judge DIR/review.png, made earlier

Needs ComfyUI, Blender and Ollama running, and a vision model in forge.json's "llm" block (README.md).

  1. plan     the model turns the brief into a job: name, subject wording, triangle budget, cut-out fill,
              and a short list of things to check later
  2. picture  FLUX draws --seeds candidates; the model picks the best, or rewrites the subject if none fit
  3. mesh     the usual pipeline: cut-out, Hunyuan3D 2, Blender low poly, review sheet (asset_pipeline.py)
  4. judge    the model looks at review.png against the brief and its checks; on a fail it changes the
              subject (new pictures) or the triangle budget and fill (new mesh), up to --rounds rounds.
              A pass needs the model's own pass and a score of at least --pass (out of 10)

Everything lands in <out>/props/<name>/ as usual, plus candidates/ and director.json (every plan, pick and
verdict). Nothing is copied into a site: a person still looks at review.png and decides.

forge.json's "llm" block may hold "banned": a list of words (brand or character names, say) that must never
reach a prompt. A plan that uses one is sent back.
"""
import json
import re
import shutil
import sys
from pathlib import Path

import asset_pipeline as ap
import comfy_run
import config
import llm

RULES = """You plan 3D props for a stylised miniature house. The pipeline: an image model draws the object from the
front on white, a mesh model turns that picture into a 3D model, and it is reduced to low poly.
Write the subject as a short plain description of ONE object (under 35 words): its shape, parts, colours and
materials. A style wrapper adds the rendering words (view, lighting, background), so do not repeat them.
What works: chunky, solid, rounded parts. What fails: thin parts (wires, cords, slim legs, fine handles),
see-through glass, text, and anything that only reads from the side. Make thin parts thick on purpose.
Never name a brand, product line, company, franchise, character or real person, and never ask to copy an
existing design: describe shapes and colours instead. Our own designs only."""

PLAN = {"type": "object", "required": ["name", "subject", "tris", "fill", "checks"], "properties": {
    "name": {"type": "string", "description": "file name, lowercase words joined by dashes"},
    "subject": {"type": "string"},
    "tris": {"type": "integer", "minimum": 600, "maximum": 6000},
    "fill": {"type": "boolean", "description": "true when the object has pale or see-through panels the cut-out could punch out"},
    "checks": {"type": "array", "items": {"type": "string"}, "maxItems": 6,
               "description": "short things to look for in the finished model, from the brief"}}}

PICK = {"type": "object", "required": ["scores", "best", "all_bad", "subject"], "properties": {
    "scores": {"type": "array", "items": {"type": "object", "required": ["image", "score", "problems"], "properties": {
        "image": {"type": "integer"}, "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "problems": {"type": "string", "maxLength": 200}}}},
    "best": {"type": "integer", "description": "1-based number of the best picture"},
    "all_bad": {"type": "boolean", "description": "true when no picture is usable"},
    "subject": {"type": "string", "maxLength": 300, "description": "when all_bad: a rewritten subject that avoids the problems; else empty"}}}

VERDICT = {"type": "object", "required": ["score", "passed", "problems", "redo", "subject", "tris", "fill"], "properties": {
    "score": {"type": "integer", "minimum": 0, "maximum": 10},
    "passed": {"type": "boolean"},
    "problems": {"type": "array", "items": {"type": "string", "maxLength": 200}, "maxItems": 6},
    "redo": {"type": "string", "enum": ["none", "picture", "mesh"],
             "description": "picture: the drawing itself is wrong (shape, colour, parts); mesh: the drawing was fine "
                            "but the 3D model is lumpy, holed or too coarse"},
    "subject": {"type": "string", "maxLength": 300, "description": "for redo picture: the rewritten subject; else empty"},
    "tris": {"type": "integer", "description": "triangle budget for the next try (same as now if unchanged)"},
    "fill": {"type": "boolean"}}}


def banned(text):
    words = [w for w in config.LLM.get("banned", []) if w]
    return [w for w in words if re.search(r"\b" + re.escape(w) + r"\b", text, re.I)]


def plan(brief, style):
    user = f"Brief: {brief}\nStyle: {style.get('about', '')}\nPlan the job."
    for _ in range(3):
        p = llm.ask(RULES, user, PLAN)
        hit = banned(p["subject"] + " " + p["name"])
        if not hit:
            p["name"] = re.sub(r"[^a-z0-9]+", "-", p["name"].lower()).strip("-") or "prop"
            return p
        user += f"\nYour last subject used a word that is not allowed ({', '.join(hit)}). Describe it without it."
    raise SystemExit(f"the planner kept using banned words: {hit}")


def pick(brief, job, files):
    user = (f"Brief: {brief}\nSubject drawn: {job['subject']}\nThese {len(files)} pictures are candidates, numbered 1 to "
            f"{len(files)} in order. Each must show exactly one object that fits the brief, whole, straight from the "
            "front, on white, with chunky parts a mesh model can rebuild. Compare them against each other and the brief "
            "(colours, which part is which colour, shape). Give every picture a different score; no ties.")
    return llm.ask(RULES, user, PICK, images=files, temperature=0.2)


def judge(brief, job, sheet):
    user = (f"Brief: {brief}\nSubject drawn: {job['subject']}\nTriangle budget: {job['tris']}\n"
            f"Checks: {'; '.join(job.get('checks', [])) or 'none'}\n"
            "This sheet shows the finished low-poly 3D model from four sides. The back is always a guess and a little "
            "blurry, so do not fail it for that. Fail it for: wrong shape or colours for the brief, missing or merged "
            "parts, lumps, holes, spikes, a melted look, or parts too thin to read. Pass it when it would look right as "
            "a small prop in a toy house. Score it 0-10. Keep each problem to one short sentence. Leave subject empty "
            "unless redo is picture.")
    return llm.ask(RULES, user, VERDICT, images=[sheet], temperature=0.2)


def draw(job, d, rnd, seeds, base_seed):
    cand = d / "candidates"; cand.mkdir(parents=True, exist_ok=True)
    comfy_run.OUT = cand
    prompt = ap.STYLES[job["style"]]["prompt"].format(subject=job["subject"])
    files = []
    for k in range(seeds):
        seed = base_seed * 100 + rnd * 10 + k
        comfy_run.run({"name": f"r{rnd}-s{seed}", "model": "flux", "seed": seed, "prompt": prompt})
        files.append(cand / f"r{rnd}-s{seed}.png")
    return files


def main():
    args = sys.argv[1:]
    opt = lambda k, d: args[args.index(k) + 1] if k in args else d  # noqa: E731
    style_name = opt("--style", ap.DEFAULT_STYLE)
    style = ap.STYLES[style_name]
    if "--judge" in args:
        d = Path(opt("--judge", "")); brief = [a for a in args if not a.startswith("--") and a != str(d)][-1]
        log = json.loads((d / "director.json").read_text()) if (d / "director.json").exists() else {}
        job = log.get("job") or {"subject": brief, "tris": "unknown", "checks": []}
        print(json.dumps(judge(brief, job, d / "review.png"), indent=2))
        return
    skip = {opt(k, None) for k in ("--style", "--name", "--rounds", "--seeds", "--seed", "--pass")}
    brief = " ".join(a for a in args if not a.startswith("--") and a not in skip)
    rounds, seeds, base_seed = int(opt("--rounds", 3)), int(opt("--seeds", 3)), int(opt("--seed", 1))
    pass_mark = int(opt("--pass", 8))

    job = plan(brief, style) | {"style": style_name}
    job["name"] = opt("--name", job["name"])
    d = ap.OUT / job["name"]; d.mkdir(parents=True, exist_ok=True)
    log = {"brief": brief, "job": dict(job), "rounds": []}
    save = lambda: (d / "director.json").write_text(json.dumps(log, indent=2))  # noqa: E731
    print(f"plan     {job['name']}: {job['subject']}  (tris {job['tris']}, fill {job['fill']})", flush=True)
    save()

    redo, verdict = "picture", None
    for rnd in range(rounds):
        entry = {"round": rnd + 1, "subject": job["subject"], "tris": job["tris"], "fill": job["fill"]}
        log["rounds"].append(entry)
        asset = {"name": job["name"], "prompt": job["subject"], "style": style_name, "tris": job["tris"], "fill": job["fill"]}
        if redo == "picture":
            files = draw(job, d, rnd + 1, seeds, base_seed)
            choice = pick(brief, job, files); entry["pick"] = choice; save()
            print(f"pick     round {rnd + 1}: " + ", ".join(f"{s['image']}={s['score']}" for s in choice["scores"])
                  + (" -- none usable" if choice["all_bad"] else f" -> {choice['best']}"), flush=True)
            if choice["all_bad"] and rnd + 1 < rounds and choice["subject"] and not banned(choice["subject"]):
                job["subject"] = choice["subject"]; continue
            best = files[max(1, min(len(files), choice["best"])) - 1]
            shutil.copy(best, d / "source.png"); entry["source"] = best.name
        ap.step_mesh(asset)
        entry["lowpoly"] = ap.step_blender(asset)
        verdict = judge(brief, job, d / "review.png"); entry["verdict"] = verdict; save()
        shutil.copy(d / "review.png", d / "candidates" / f"review-r{rnd + 1}.png")
        # the model is lenient: it passes things it has just listed problems for, so the bar is set here
        verdict["passed"] = verdict["passed"] and verdict["score"] >= pass_mark
        print(f"judge    round {rnd + 1}: {verdict['score']}/10 {'PASS' if verdict['passed'] else 'FAIL'}"
              + (f" -- {'; '.join(verdict['problems'])}" if verdict["problems"] else ""), flush=True)
        entry["verdict"] = verdict; save()
        if verdict["passed"]:
            break
        redo = verdict["redo"] if verdict["redo"] != "none" else "picture"  # a fail always tries something
        if redo == "picture" and verdict["subject"] and not banned(verdict["subject"]):
            job["subject"] = verdict["subject"]
        job["tris"] = max(600, min(6000, verdict.get("tris") or job["tris"]))
        job["fill"] = bool(verdict.get("fill", job["fill"]))
    log["result"] = {"passed": bool(verdict and verdict["passed"]), "glb": str(d / f"{job['name']}.glb"),
                     "review": str(d / "review.png")}
    save()
    print(f"done     {'passed' if log['result']['passed'] else 'NOT passed'}: {log['result']['glb']}\n"
          f"         look at {log['result']['review']} before using it")


if __name__ == "__main__":
    main()
