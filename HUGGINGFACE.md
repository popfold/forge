# Hugging Face: trying 3D models without installing them

The forge runs everything locally. To try a 3D model before downloading and installing it (or one that needs more
than a 12 GB GPU, or Linux), you can run it on Hugging Face instead, from the same picture the forge drew. This page
covers how that works, what it costs, and how to do it yourself.

## Three different things on Hugging Face

| What | What it is | Who pays for the GPU |
|---|---|---|
| **Models** (`huggingface.co/<org>/<model>`) | The files: weights, a model card, a licence. Downloading them is free. | You, on your own GPU |
| **Spaces** (`huggingface.co/spaces/<org>/<name>`) | A small app (usually Gradio) someone runs on Hugging Face's machines: a web page, and the same functions as an API | See below: usually **ZeroGPU** |
| **Inference Providers** | Hosted endpoints for chat, image and other tasks, sold through Hugging Face by partner clouds | You, per call, from your credit balance |

Most 3D models (TRELLIS.2, Hunyuan3D 2.1, Pixal3D, Hi3D, Direct3D-V3, TripoSG ...) are available only as
**Spaces**. Inference Providers do not offer image-to-3D.

## How a ZeroGPU Space is charged

A Space shows its hardware on its page and in the API (`runtime.hardware.current`). Most 3D demos use
`zero-a10g`, which is **ZeroGPU**: a shared pool of large GPUs. A GPU is attached to the Space only while one of its
functions runs, and the time it is attached is charged to **the account that called it**, not to the Space's owner.

Each account has a daily allowance (the current numbers are on
[the ZeroGPU page](https://huggingface.co/docs/hub/spaces-zerogpu)):

| Account | Included GPU time a day | Queue |
|---|---|---|
| Not signed in | 2 minutes | Low |
| Free | 5 minutes | Medium |
| PRO | 40 minutes | Highest |

- The allowance resets 24 hours after the first GPU use of the day.
- **Beyond it, a PRO account is billed automatically** at $1 per 10 minutes of GPU time, against prepaid credits,
  if it has any. With no credits, the Space refuses with a "GPU quota exceeded" error. Check your balance in
  [billing settings](https://huggingface.co/settings/billing) before a long batch.
- Only the time the GPU is attached counts, not uploading or queueing. A 3D model typically takes 1 to 2 minutes
  a run, so 40 minutes covers about 20 to 30 runs a day.
- A Space on paid hardware of its own (for example `a100-large`) is paid for by its owner. A Space on
  `cpu-basic` that makes 3D models is usually calling a paid service of its own.

How to tell which applies to your account: `GET https://huggingface.co/api/whoami-v2` with your token reports
`isPro` and `canPay`. A Space's hardware is in `GET https://huggingface.co/api/spaces/<id>?expand[]=runtime`.

## Calling a Space from a script

Every Gradio Space is also an API. `gradio_client` (from Hugging Face, `pip install gradio_client`) prints the API
and calls it. Pass your token so the time counts against your account (and its larger allowance), not as an
anonymous visitor:

```python
from gradio_client import Client, handle_file
c = Client("tencent/Hunyuan3D-2.1", token=open("<your token file>").read().strip())
c.view_api()                                   # every function, its inputs and outputs
r = c.predict(handle_file("prepared.png"), None, None, None, None, 30, 5.0, 1234, 256, True, 8000, False,
              api_name="/generation_all")      # files come back to a local temp folder
```

Things that tripped us up:

- **Inputs are typed exactly.** Hi3D's resolution is the number `1024`. The string `"1024"` is refused.
- **Some outputs are `gr.update` dicts** (`{"value": path, "__type__": "update"}`), not paths: take `["value"]`.
- **Some Spaces return files the client may not download** (Pixal3D's preview renders give 403). Make a second
  `Client(..., download_files=False)` for that call and pass on the server path it returns.
- **Two-step Spaces keep state on the server.** Pixal3D returns a `state_path`, and a second call turns it into a
  GLB. Use the same Space for both calls.
- **Results are dense.** Hi3D gives about a million triangles and no colours. Direct3D-V3 gives 200,000, also with
  no colours. Reduce them (`blender_fit.py --tris`) and paint them before using them as props.
- A Space can be paused, rebuilding or broken on any day. `stage` in the API says `RUNNING` when it can be called.

## What we tried (two forge props, the same cut-out picture)

| Model | Space | Gives | Seconds a prop | Notes |
|---|---|---|---|---|
| Pixal3D (TencentARC, MIT) | `TencentARC/Pixal3D` | shape + texture, reduced to the triangles asked | 100-120 | Dented the desk at 3,000 triangles; texture good |
| Hunyuan3D 2.1 (Tencent, community licence) | `tencent/Hunyuan3D-2.1` | shape + PBR texture, 40,000 triangles | 110-130 | Very good figure; the desk came with a floor slab |
| Hi3D v1.0 (Sparc3D refiner) | `hugging-apps/hi3dv1` | shape only, ~1M triangles | 80-130 | The sharpest shapes: drawers, knobs, scales |
| Direct3D-V3 (MIT) | `hugging-apps/direct3d-v3` | shape only, refines a mesh you give it | 60-75 | Sharp, and keeps the forge mesh's pose |

## Doing it yourself, without an agent

- **The Space's own page.** Open it signed in to Hugging Face, drop in a picture, press Generate, download the
  GLB. The time counts against your allowance, as it does from a script.
- **The Hugging Face MCP server** ([huggingface.co/mcp](https://huggingface.co/settings/mcp)) can offer chosen
  Spaces as tools to any MCP client: HuggingChat, or an agent such as OpenCode running an open model through
  Inference Providers.
- **Locally, with a desktop app.** ComfyUI (already used by the forge) has built-in templates for TRELLIS.2 and
  Hunyuan3D in its own UI at `http://127.0.0.1:8188`. Pinokio installs many 3D demos locally with one click.
  Neither one uses Hugging Face's GPUs.
