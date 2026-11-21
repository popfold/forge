"""A local language model through Ollama's HTTP API, for the planner and judge in forge.py.

    import llm
    reply = llm.ask(system, user, schema, images=[path, ...])   # a dict that matches the JSON schema

The model and address come from config.LLM (forge.json "llm" block, or OLLAMA_URL / POPFOLD_LLM_MODEL).
A vision model (for example Qwen3-VL) can do every role; a text-only model can only plan.

Repeatable: every call is sampled with a fixed seed at temperature 0, and every answer is kept in
<out>/llm-cache/, keyed by the model's digest, the prompts, the schema and the pictures' bytes. Asking the
same question again returns the kept answer without loading the model, so a run made twice decides the same
way twice. Pass cache=False to ask afresh.

The GPU is shared with ComfyUI, so a call that does reach the model frees ComfyUI's models first and asks
Ollama to unload the language model as soon as it answers (keep_alive 0). Slower than keeping both loaded,
but a 12 GB card cannot hold FLUX or Hunyuan3D and an 8B model at once.
"""
import base64
import hashlib
import json
import urllib.request
from pathlib import Path

import config

CACHE = config.OUT / "llm-cache"
_digest = {}


def _post(url, data, timeout):
    req = urllib.request.Request(url, data=json.dumps(data).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body.strip() else {}


def free_comfy():
    """Ask ComfyUI to drop its models from the GPU. Harmless if ComfyUI is not running."""
    try:
        _post(config.COMFY_URL + "/free", {"unload_models": True, "free_memory": True}, 30)
    except Exception:
        pass


def model():
    m = config.LLM.get("model")
    if not m:
        raise SystemExit("No language model set: put \"llm\": {\"model\": ...} in forge.json or set POPFOLD_LLM_MODEL.")
    return m


def digest():
    """The model file's digest, so a re-pulled or different model never answers from another's cache."""
    m = model()
    if m not in _digest:
        with urllib.request.urlopen(config.LLM["url"] + "/api/tags", timeout=10) as r:
            tags = json.loads(r.read())["models"]
        found = [t["digest"] for t in tags if t["name"] == m or t.get("model") == m]
        if not found:
            raise SystemExit(f"Ollama does not have {m}. Pull it first (README.md).")
        _digest[m] = found[0]
    return _digest[m]


def ask(system, user, schema, images=(), temperature=0.0, seed=1, cache=True, timeout=600):
    pics = [Path(p).read_bytes() for p in images]
    key = hashlib.sha256(json.dumps({
        "digest": digest(), "system": system, "user": user, "schema": schema, "temperature": temperature,
        "seed": seed, "images": [hashlib.sha256(b).hexdigest() for b in pics]}, sort_keys=True).encode()).hexdigest()
    kept = CACHE / f"{key[:32]}.json"
    if cache and kept.exists():
        return json.loads(kept.read_text(encoding="utf-8"))["answer"]
    free_comfy()
    msg = {"role": "user", "content": user}
    if pics:
        msg["images"] = [base64.b64encode(b).decode() for b in pics]
    r = _post(config.LLM["url"] + "/api/chat", {
        "model": model(), "stream": False, "keep_alive": 0, "format": schema,
        "options": {"temperature": temperature, "seed": seed, "num_ctx": 8192, "num_predict": 1500},
        "messages": [{"role": "system", "content": system}, msg],
    }, timeout)
    text = r["message"]["content"]
    try:
        answer = json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"model did not return JSON: {text[:500]}") from e
    CACHE.mkdir(parents=True, exist_ok=True)
    kept.write_text(json.dumps({"model": model(), "user": user, "answer": answer}, indent=2), encoding="utf-8")
    return answer
