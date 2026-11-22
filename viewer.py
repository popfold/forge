"""A small web viewer for every model the forge made: a grid of thumbnails, tap one to turn it around.

    python viewer.py [--port 5195] [--host 127.0.0.1]

Serves <out> (config.py) read-only: the library in <out>/models, each prop's final model, review sheet and
progress pictures in <out>/props, and the test meshes in <out>/meshes. Each model is shown with what its
run.json says about it (the brief, the judge's score and problems, the front match, triangles, size).
--host 0.0.0.0 lets a phone on the same network open it; a tunnel to the port works too (what comes through a
tunnel can only read: it cannot save thumbnails). three.js comes from $FORGE_THREE, a node_modules here,
or a Popfold checkout beside this folder. Standard library only.

Thumbnails are drawn by the first browser that shows a model and sent back to <out>/viewer-thumbs, so a phone
downloads a small picture per model instead of every model.
"""
import json
import mimetypes
import os
import re
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import config

HERE = Path(__file__).parent
# three.js: $FORGE_THREE, else a node_modules here (npm/pnpm add three), else a Popfold checkout beside this one
THREE_AT = ([Path(os.environ["FORGE_THREE"])] if os.environ.get("FORGE_THREE") else []) + [
    HERE / "node_modules" / "three", HERE.parent / "popfold" / "node_modules" / "three"]
THREE = next((p for p in THREE_AT if (p / "build" / "three.module.js").exists()), THREE_AT[0])
OUT = config.OUT
THUMBS = OUT / "viewer-thumbs"
SERVED = {".glb", ".png", ".jpg", ".jpeg", ".webp", ".json"}
# who made a model: a short name for the card, and the whole chain for the info panel
FORGE = "{painter} drew it, {mesh} made the 3D model, Blender reduced it; Qwen3-VL 8B planned and judged (all local)"
PAINTER = {"flux": "FLUX schnell", "chroma": "Chroma1-HD"}
MESHER = {"trellis": "TRELLIS.2", "hunyuan": "Hunyuan3D 2", "hunyuan21": "Hunyuan3D 2.1"}
# compare/<way>/: short name, whole chain, and where it ran ("remote" ones may cost credits). A log.json "maker",
# "made_by", "where" and "credits" say it better, when there are
WAYS = {
    "rodin": ("Rodin", "Hyper3D Rodin (hosted, paid credits), from the forge's picture", "remote"),
    "blender": ("Blender", "the forge's TRELLIS.2 model, cleaned up and reduced in Blender by a script", "local"),
    "pixal3d": ("Pixal3D", "Pixal3D (TencentARC), shape and texture, on a Hugging Face Space", "remote"),
    "hunyuan21": ("Hunyuan3D 2.1", "Hunyuan3D 2.1 (Tencent), shape and PBR texture, on a Hugging Face Space", "remote"),
    "hi3d": ("Hi3D", "Hi3D v1.0 (a Sparc3D refiner), shape only, on a Hugging Face Space", "remote"),
    "hy21-local": ("Hunyuan3D 2.1", "Hunyuan3D 2.1 in ComfyUI on this machine (shape), the picture projected on in Blender", "local"),
    "direct3d": ("Direct3D-V3", "Direct3D-V3, shape only, refining the forge's TRELLIS.2 mesh, on a Hugging Face Space", "remote"),
}
PAID_WAYS = {"rodin"}  # a paid service: counted as paid when its log has no credits figure
ROUND = re.compile(r"^(?P<name>.+)-r(?P<round>\d+)-s(?P<seed>\d+)(?P<dense>-dense)?\.glb$")


def runs():
    """Every prop's run.json, by name."""
    found = {}
    for f in sorted((OUT / "props").glob("*/run.json")):
        try:
            found[f.parent.name] = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return found


def rel(p: Path):
    return p.relative_to(OUT).as_posix()


def catalogue():
    """One entry per model file, newest first, with what the forge recorded about it."""
    all_runs = runs()
    items = []

    def entry(f: Path, group, **extra):
        st = f.stat()
        thumb = f"{rel(f).replace('/', '__')}-{int(st.st_mtime)}.webp"
        return {"file": rel(f), "group": group, "bytes": st.st_size, "mtime": int(st.st_mtime),
                "thumb": thumb, "has_thumb": (THUMBS / thumb).exists(), "where": "local", "paid": False, **extra}

    for f in sorted((OUT / "models").glob("*.glb")):
        m = ROUND.match(f.name)
        name = m["name"] if m else f.stem
        run = all_runs.get(name, {})
        mesher = MESHER.get(run.get("mesh", "trellis"), run.get("mesh"))
        info = {"name": name, "brief": run.get("brief", ""), "style": run.get("style", ""), "maker": mesher,
                "made_by": f"{mesher}, its dense model before Blender" if m and m["dense"]
                else FORGE.format(mesh=mesher, painter=PAINTER.get(run.get("painter", "flux"), "FLUX schnell"))}
        if m:
            info |= {"round": int(m["round"]), "seed": int(m["seed"]), "dense": bool(m["dense"])}
            r = next((r for r in run.get("round", []) if r.get("library") == f.name.replace("-dense", "")), None)
            if r:
                v = r.get("verdict") or {}
                info |= {"subject": r.get("subject", ""), "passed": bool(r.get("passed")), "score": v.get("score"),
                         "problems": v.get("problems", []), "gate_fails": r.get("gate_fails", []),
                         "front_iou": (r.get("mesh") or {}).get("front_iou"), "tris": (r.get("mesh") or {}).get("tris")}
            res = run.get("result") or {}
            info["final"] = bool(res.get("passed")) and res.get("round") == info["round"] and not info["dense"]
        items.append(entry(f, "library", **info))
    for d in sorted((OUT / "props").iterdir()) if (OUT / "props").is_dir() else []:
        glb = d / f"{d.name}.glb"
        if not glb.exists():
            continue
        run = all_runs.get(d.name, {})
        res = run.get("result") or {}
        pics = sorted(rel(p) for p in (d / "progress").glob("*.jpg")) if (d / "progress").is_dir() else []
        last = next((r for r in run.get("round", []) if r.get("round") == res.get("round")), {})
        v, mesh = last.get("verdict") or {}, last.get("mesh") or {}
        mesher = MESHER.get(run.get("mesh", "trellis"), run.get("mesh"))
        items.append(entry(glb, "props", name=d.name, brief=run.get("brief", ""), style=run.get("style", ""),
                           maker=mesher, made_by=FORGE.format(mesh=mesher, painter=PAINTER.get(run.get("painter", "flux"), "FLUX schnell")),
                           round=res.get("round"), subject=last.get("subject"), score=v.get("score"),
                           problems=v.get("problems", []), front_iou=mesh.get("front_iou"), tris=mesh.get("tris"),
                           passed=res.get("passed"), review=rel(d / "review.png") if (d / "review.png").exists() else None,
                           progress=pics))
    # side-by-side trials: <out>/compare/<way>/<prop>/<prop>.glb, with its review sheet and log.json
    for f in sorted((OUT / "compare").glob("*/*/*.glb")) if (OUT / "compare").is_dir() else []:
        log = f.parent / "log.json"
        try:
            log = json.loads(log.read_text(encoding="utf-8")) if log.exists() else {}
        except ValueError:
            log = {}
        way = f.parents[1].name
        review = f.parent / "review.png"
        prop = f.parent.name
        short, long, where = WAYS.get(way, (way, way, "local"))
        where = log.get("where", where)
        credits = log.get("credits")
        paid = where == "remote" and (credits > 0 if isinstance(credits, (int, float)) else way in PAID_WAYS)
        items.append(entry(f, "compare", name=f"{prop} · {way}" + ("" if f.stem == prop else f" · {f.stem}"), brief=all_runs.get(log.get("prop", ""), {}).get("brief", ""),
                           tris=log.get("triangles"), subject=f"{way}: {log.get('wall_seconds')} s"
                           + (f", {log['credits']} credits" if log.get("credits") is not None else ""),
                           problems=[log["notes"]] if isinstance(log.get("notes"), str) else log.get("notes", []),
                           review=rel(review) if review.exists() else None,
                           maker=log.get("maker", short), made_by=log.get("made_by", long), where=where, paid=paid,
                           credits=credits))
    for f in sorted((OUT / "meshes").glob("*.glb")) if (OUT / "meshes").is_dir() else []:
        maker = next((v for k, v in (("trellis", "TRELLIS.2"), ("hunyuan", "Hunyuan3D 2")) if k in f.stem), None)
        items.append(entry(f, "tests", name=f.stem, maker=maker, made_by=maker and f"{maker} (an early test)"))
    items.sort(key=lambda i: -i["mtime"])
    return items


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body: bytes, kind: str, cache="no-cache"):
        self.send_response(200)
        self.send_header("Content-Type", kind); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers(); self.wfile.write(body)

    def file(self, root: Path, sub: str, cache="no-cache"):
        p = (root / sub).resolve()
        if root.resolve() not in p.parents or not p.is_file():
            return self.send_error(404)
        self.send(p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream", cache)

    def do_GET(self):
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            return self.file(HERE, "viewer.html")
        if path == "/api/models":
            return self.send(json.dumps(catalogue()).encode(), "application/json")
        if path.startswith("/three/"):
            return self.file(THREE, path[7:], "max-age=86400")
        if path.startswith("/thumbs/"):
            return self.file(THUMBS, path[8:], "max-age=31536000")
        if path.startswith("/files/") and Path(path).suffix.lower() in SERVED:
            return self.file(OUT, path[7:])
        self.send_error(404)

    def do_POST(self):  # a thumbnail drawn by a browser: <file with / as __>-<mtime>.webp, nothing else
        name = unquote(urlparse(self.path).path)[8:]
        n = int(self.headers.get("Content-Length", 0))
        if self.headers.get("Cf-Connecting-IP") or self.headers.get("X-Forwarded-For"):  # through a tunnel: read only
            return self.send_error(403)
        if not self.path.startswith("/thumbs/") or not re.fullmatch(r"[\w.\-]+\.webp", name) or not 0 < n < 2_000_000:
            return self.send_error(400)
        THUMBS.mkdir(parents=True, exist_ok=True)
        (THUMBS / name).write_bytes(self.rfile.read(n))
        self.send(b"ok", "text/plain")


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d: type(d)(a[a.index(k) + 1]) if k in a else d  # noqa: E731
    host, port = opt("--host", "127.0.0.1"), opt("--port", 5195)
    if not (THREE / "build" / "three.module.js").exists():
        sys.exit("three.js not found: set FORGE_THREE to a three folder, run pnpm add three here, "
                 "or keep a Popfold checkout (with pnpm install done) beside this folder")
    print(f"forge viewer: http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}  ({OUT})", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
