"""Settings for the forge scripts: where ComfyUI, Blender and the language model are, and where output goes.

Order of precedence, highest first:
  1. environment variables   COMFY_URL, POPFOLD_BLENDER, POPFOLD_FORGE_OUT, OLLAMA_URL, POPFOLD_LLM_MODEL,
                             RODIN_API_KEY, RODIN_KEY_FILE
  2. a settings file         the path in POPFOLD_FORGE_CONFIG, else ./forge.json, else forge.json beside this file
  3. the defaults below

forge.json (copy forge.example.json) is for one machine and is not committed.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).parent
DEFAULTS = {
    "comfy_url": "http://127.0.0.1:8188",   # the ComfyUI server's HTTP API
    "blender": "blender",                   # the Blender program (a full path, or a name on PATH)
    "out": "forge-out",                     # where pictures, meshes and finished props are written
    # a local vision language model (Ollama) for director.py's planner and judge; "banned": words kept out of prompts
    "llm": {"url": "http://127.0.0.1:11434", "model": ""},
    # Hyper3D Rodin, a paid hosted image-to-3D service (rodin.py); point key_file at a file outside the repo
    "rodin": {"key_file": "", "tier": "Gen-2.5-Medium"},
}


def load():
    cfg = json.loads(json.dumps(DEFAULTS))
    named = os.environ.get("POPFOLD_FORGE_CONFIG")
    path = Path(named) if named else next((p for p in (Path.cwd() / "forge.json", HERE / "forge.json") if p.exists()), None)
    if path:
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg["llm"].update(user.pop("llm", {}))
        cfg["rodin"].update(user.pop("rodin", {}))
        cfg.update(user)
    env = os.environ
    cfg["comfy_url"] = env.get("COMFY_URL", cfg["comfy_url"]).rstrip("/")
    cfg["blender"] = env.get("POPFOLD_BLENDER", cfg["blender"])
    cfg["out"] = env.get("POPFOLD_FORGE_OUT", cfg["out"])
    cfg["llm"]["url"] = env.get("OLLAMA_URL", cfg["llm"]["url"]).rstrip("/")
    cfg["llm"]["model"] = env.get("POPFOLD_LLM_MODEL", cfg["llm"]["model"])
    cfg["rodin"]["key"] = env.get("RODIN_API_KEY", "")
    cfg["rodin"]["key_file"] = env.get("RODIN_KEY_FILE", cfg["rodin"]["key_file"])
    return cfg


CFG = load()
COMFY_URL = CFG["comfy_url"]
BLENDER = CFG["blender"]
OUT = Path(CFG["out"]).resolve()
LLM = CFG["llm"]
RODIN = CFG["rodin"]


def styles():
    """Every style: the files in ./styles (a site's own) on top of the ones that ship here."""
    found = {}
    for folder in (HERE / "styles", Path.cwd() / "styles"):
        for f in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            found[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return found
