"""A local language model through Ollama's HTTP API, for the planner and judge roles in director.py.

    import llm
    reply = llm.ask(system, user, schema, images=[path, ...])   # a dict that matches the JSON schema

The model and address come from config.LLM (forge.json "llm" block, or OLLAMA_URL / POPFOLD_LLM_MODEL).
A vision model (for example Qwen3-VL) can do both roles; a text-only model can only plan.

The GPU is shared with ComfyUI, so every call frees ComfyUI's models first and asks Ollama to unload the
language model as soon as it answers (keep_alive 0). Slower than keeping both loaded, but a 12 GB card
cannot hold FLUX or Hunyuan3D and an 8B model at once.
"""
import base64
import json
import urllib.request
from pathlib import Path

import config


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


def ask(system, user, schema, images=(), temperature=0.4, timeout=600):
    model = config.LLM.get("model")
    if not model:
        raise SystemExit("No language model set: put \"llm\": {\"model\": ...} in forge.json or set POPFOLD_LLM_MODEL.")
    free_comfy()
    msg = {"role": "user", "content": user}
    if images:
        msg["images"] = [base64.b64encode(Path(p).read_bytes()).decode() for p in images]
    r = _post(config.LLM["url"] + "/api/chat", {
        "model": model, "stream": False, "keep_alive": 0, "format": schema,
        "options": {"temperature": temperature, "num_ctx": 8192, "num_predict": 1500},
        "messages": [{"role": "system", "content": system}, msg],
    }, timeout)
    text = r["message"]["content"]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"model did not return JSON: {text[:500]}") from e
