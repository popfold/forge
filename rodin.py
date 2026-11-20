"""Picture or text to a textured 3D model with Hyper3D Rodin, a paid hosted service (needs an API key).

    python rodin.py IMAGE.png [IMAGE2.png ...] --out DIR [--prompt "..."] [--tier Gen-2.5-Medium] [--faces 20000]
    python rodin.py --text "a chunky toy armchair" --out DIR
    python rodin.py --balance

Writes everything Rodin returns (the .glb, its textures, a preview) into DIR, plus rodin.json (the request,
the task id and the credits used). Up to five pictures of the same object; the first one sets the colours.

The key: RODIN_API_KEY, or a file named by RODIN_KEY_FILE or forge.json's "rodin": {"key_file": ...}. Never
put the key itself in a tracked file. Each generation costs credits (about 0.5 at Gen-2.5 tiers below
Extreme-High; see the Hyper3D pricing page), so ask before running a batch. Pictures leave this machine: never
send a picture of a real person without that person's consent.
"""
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import config

API = "https://api.hyper3d.com/api/v2"


def _key():
    r = config.RODIN
    if r.get("key"):
        return r["key"]
    if r.get("key_file"):
        return Path(r["key_file"]).read_text(encoding="utf-8").strip()
    raise SystemExit("No Rodin key: set RODIN_API_KEY, or RODIN_KEY_FILE / forge.json \"rodin\": {\"key_file\": ...}.")


def _call(path, body=None, files=(), method="POST", timeout=120):
    headers = {"Authorization": "Bearer " + _key()}
    data = None
    if files or (body is not None and method == "POST" and path == "/rodin"):
        # multipart: every field as text, each picture as an "images" file part
        b = uuid.uuid4().hex; parts = []
        for k, v in (body or {}).items():
            for item in (v if isinstance(v, list) else [v]):
                parts.append(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{item}\r\n'.encode())
        for f in files:
            f = Path(f); ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            parts.append(f'--{b}\r\nContent-Disposition: form-data; name="images"; filename="{f.name}"\r\n'
                         f'Content-Type: {ctype}\r\n\r\n'.encode() + f.read_bytes() + b"\r\n")
        data = b"".join(parts) + f"--{b}--\r\n".encode()
        headers["Content-Type"] = f"multipart/form-data; boundary={b}"
    elif body is not None:
        data = json.dumps(body).encode(); headers["Content-Type"] = "application/json"
    for attempt in range(5):
        req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code >= 500:  # throttled or a server hiccup: wait and try again
                time.sleep(float(e.headers.get("Retry-After") or 5 * (attempt + 1))); continue
            raise RuntimeError(f"Rodin {path}: HTTP {e.code} {e.read()[:300]!r}") from e
        # Rodin answers errors with a 201 too; the body says so
        if isinstance(out, dict) and out.get("error"):
            raise RuntimeError(f"Rodin {path}: {out['error']} {out.get('message', '')}")
        return out
    raise RuntimeError(f"Rodin {path}: still throttled after 5 tries")


def balance():
    return _call("/check_balance", method="GET")["balance"]


def generate(out, images=(), prompt=None, tier=None, faces=None, fmt="glb", deadline=1800):
    """Submit, wait, download. Returns the paths written."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    body = {"tier": tier or config.RODIN.get("tier", "Gen-2.5-Medium"), "geometry_file_format": fmt, "mesh_mode": "Raw"}
    if prompt:
        body["prompt"] = prompt
    if faces:
        body["quality_override"] = int(faces)
    t0 = time.time()
    sub = _call("/rodin", body, files=images)
    log = {"request": body, "images": [str(i) for i in images], "task": sub["uuid"], "credits": sub.get("consumed")}
    (out / "rodin.json").write_text(json.dumps(log, indent=2))
    print(f"submitted {sub['uuid']} ({sub.get('consumed')} credits)", flush=True)
    wait, last = 5, None
    while True:
        time.sleep(wait); wait = min(30, wait * 1.5)
        jobs = _call("/status", {"subscription_key": sub["jobs"]["subscription_key"]})["jobs"]
        states = sorted({j["status"] for j in jobs})
        if states != last:
            print(f"  {int(time.time() - t0):>4}s  {', '.join(states)}", flush=True); last = states
        if "Failed" in states:
            raise RuntimeError(f"Rodin task {sub['uuid']} failed")
        if states == ["Done"]:
            break
        if time.time() - t0 > deadline:
            raise RuntimeError(f"Rodin task {sub['uuid']} not done after {deadline} s")
    written = []
    for item in _call("/download", {"task_uuid": sub["uuid"]})["list"]:
        dest = out / Path(item["name"]).name  # names can carry folders; keep the file name only
        with urllib.request.urlopen(item["url"], timeout=300) as r:
            dest.write_bytes(r.read())
        written.append(dest)
    log |= {"seconds": round(time.time() - t0), "files": [p.name for p in written]}
    (out / "rodin.json").write_text(json.dumps(log, indent=2))
    return written


if __name__ == "__main__":
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d  # noqa: E731
    if "--balance" in a:
        print(f"balance: {balance()} credits"); sys.exit()
    values = {opt(k) for k in ("--out", "--prompt", "--text", "--tier", "--faces", "--format")}
    images = [x for x in a if not x.startswith("--") and x not in values]
    if not opt("--out") or not (images or opt("--text")):
        raise SystemExit(__doc__)
    files = generate(opt("--out"), images, opt("--text") or opt("--prompt"), opt("--tier"), opt("--faces"),
                     opt("--format", "glb"))
    print("\n".join(f"wrote {f}" for f in files))
    print(f"balance now: {balance()} credits")
