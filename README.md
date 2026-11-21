# Forge: the model pipeline

Scripts that make Popfold props and wall art with local AI models and Blender. A prompt goes in, a low-poly
textured `.glb` that follows Popfold's prop convention comes out.

> **Alpha.** These are working scripts, not a polished tool. They were written on Windows with one GPU.

```
prompt ── FLUX ──▶ picture ── rembg ──▶ cut-out ── Hunyuan3D 2 ──▶ dense mesh ── Blender ──▶ low-poly .glb + review sheet
```

## What you need

| Thing | Notes |
|---|---|
| ComfyUI, running | With `flux1-schnell-fp8.safetensors` and `hunyuan3d-dit-v2.safetensors` in `models/checkpoints/` (`sd_xl_base_1.0.safetensors` too, for tileable patterns). A 12 GB GPU is enough. |
| Blender 4.2 or newer | Tested on 5.3. |
| Python 3 | With the packages in `requirements.txt`. ComfyUI's own virtual environment already has most of them. |
| Ollama (optional) | A local vision language model for `forge.py`. Tested with Qwen3-VL 8B (5.8 GB): `ollama pull hf.co/Qwen/Qwen3-VL-8B-Instruct-GGUF:Q4_K_M`, then set `llm.model` to that name. |

Check the licence of each model before you publish or sell what it makes. They differ.

## Settings

Copy `forge.example.json` to `forge.json` (not committed) and edit it, or set environment variables:

| Setting | Environment variable | Default |
|---|---|---|
| `comfy_url` | `COMFY_URL` | `http://127.0.0.1:8188` |
| `blender` | `POPFOLD_BLENDER` | `blender` (on PATH) |
| `out` | `POPFOLD_FORGE_OUT` | `forge-out` in the current folder |
| `llm.url` | `OLLAMA_URL` | `http://127.0.0.1:11434` |
| `llm.model` | `POPFOLD_LLM_MODEL` | none |

`config.py` reads them. Keep the output folder outside any site: copy only the finished `.glb` into
`<site>/public/models/props/`.

## Make props

Run from this folder, with ComfyUI running:

```
python asset_pipeline.py jobs/example-props.json                 # every asset in the file
python asset_pipeline.py jobs/example-props.json mushroom-lamp   # one asset
python asset_pipeline.py assets.json --from-image                # keep the pictures, redo mesh and Blender
python asset_pipeline.py assets.json --from-mesh                 # redo only the Blender step
python asset_pipeline.py assets.json --style=refined
```

Each asset lands in `<out>/props/<name>/`: `source.png` (the picture), `prepared.png` and `mask.png` (the cut-out),
`raw.glb` (the dense mesh), `<name>.glb` (the finished prop), `review.png` (four views) and `log.json` (timings).
**Look at `review.png` before using a prop.**

A job file is a list:

```json
[{ "name": "armchair", "prompt": "toy armchair, lilac velvet, short gold legs", "seed": 101, "tris": 2500 }]
```

| Field | Meaning |
|---|---|
| `name` | File name of the prop (lowercase, dashes) |
| `prompt` | The object. The style adds the rest |
| `seed` | Change it to get a different picture |
| `tris` | Triangle budget (default from the style) |
| `style` | A file in `styles/` (default `toy`) |
| `raw_prompt` | A complete prompt, skipping the style |
| `flat`, `depth` | For thin wall pieces |
| `fill` | Keep pale glass or panels the cut-out removed |
| `palette`, `colours` | Flat colours per part instead of the projected picture |

`pick_source.py assets.json --out=DIR --seeds=4 name` draws several pictures and keeps the most front-on one.

### Styles

A style is a JSON file in `styles/`: the prompt wrapped around every subject, and a default triangle budget.

```json
{ "about": "Chunky, glossy toy furniture.", "prompt": "single {subject}, chunky stylised toy, ... 3d render", "tris": 2500 }
```

A site can keep its own in a `styles/` folder where the script is run from; those are added to the ones here.
One style per site is what makes forty props look like one set.

## From a brief, the same way every time

`forge.py` takes a one-line brief instead of a job file and runs the whole chain with local models only: Ollama
plans and judges, ComfyUI draws and meshes, Blender reduces and renders. Nothing leaves the machine.

```
python forge.py "a sprinkled donut"
python forge.py "a chunky mint-green mushroom lamp, cream stem" --style toy --name mushroom-lamp
python forge.py "..." --seed 2 --seeds 4 --rounds 3 --pass 7
```

The script makes every decision; the model only answers questions. Each round:

| Step | Who decides | What |
|---|---|---|
| plan | model, once | Subject wording, triangle budget, things to check. Any "no …" or "without …" clause is dropped: the picture model has no negative prompt, and the picker would hold it against the picture. `--fill` keeps pale panels the cut-out punches out |
| draw | fixed seeds | `--seeds` FLUX pictures; seeds come from `--seed` and the round |
| measure | numbers | Each cut-out: one object, not cut off, 4 to 75 % of the picture, white background, symmetry. A picture that fails is out |
| pick | model, then a rule | The model scores the pictures; the best that passed the measures wins (ties: symmetry, then seed) |
| mesh | the pipeline | Cut-out, Hunyuan3D 2, Blender low poly, review sheet |
| check | numbers, then model | The mesh's front silhouette must match the picture (IoU 0.55 or more); then the model scores `review.png`. A pass needs both, and a score of `--pass` or more |
| next | a rule | A wrong drawing: the model rewrites the subject, new seeds. A poor mesh: the next-best picture and 1.5x the triangles |

The bars are `GATES` at the top of `forge.py`.

**Repeatable.** Every step is keyed by a hash of what goes into it: the prompt, the seed, the input files, the
scripts that do the work and the language model's digest. A step whose key matches what is on disk is skipped.
The model is asked at temperature 0 with a fixed seed, and `llm.py` keeps every answer in `<out>/llm-cache/`, so
the same question gets the same answer without loading the model. Running the same command twice gives the same
files, the second time in seconds; a run that stopped part way picks up where it stopped. `--fresh` redoes every
step and asks the model again. A different `--seed` gives a different prop.

It writes `<out>/props/<name>/` (`<name>` from `--name`, or the brief's first words): `r1/`, `r2/` ... with each
round's candidate pictures, cut-out, meshes and sheet, `progress/` (one labelled picture per stage, in order: the
candidates with their scores and which won, then the cut-out beside the model's sheet and verdict; it prints each
path as it writes it), the chosen `<name>.glb` and `review.png`, and `run.json`
(every input, measure, score and decision, and the versions used). When no round passes, the best round is still
kept and `run.json` says it did not pass. **Look at `review.png` before using a prop.**

`llm.banned` in `forge.json` is a list of words (brand or character names) that must never reach a prompt: a
brief that uses one is refused, and a plan or rewrite that uses one is asked again. `llm.py` is the small Ollama
client.

The GPU is shared: a model call that is not in the cache frees ComfyUI's models first, and Ollama unloads the
language model as soon as it answers.

## Use Hyper3D Rodin instead of Hunyuan3D

`rodin.py` sends one to five pictures (or a text prompt) to Hyper3D Rodin, a paid hosted service, and downloads the
textured model it makes. Its meshes are usually cleaner than Hunyuan3D's and come with a full texture, but each
one costs credits and the pictures leave the machine.

```
python rodin.py --balance                                            # free: credits left
python rodin.py <out>/props/armchair/source.png --out <out>/rodin/armchair --faces 8000
python rodin.py --text "a chunky toy armchair, lilac velvet" --out <out>/rodin/armchair
```

**API generation needs a Business or Enterprise plan.** On a lower plan the balance check works but a generation
fails with `API_NO_ACTIVE_SUBSCRIPTION`. Set the key with `RODIN_API_KEY`, or point `rodin.key_file` in
`forge.json` (or `RODIN_KEY_FILE`) at a file outside the repository. `rodin.tier` sets the default tier
(`Gen-2.5-Medium`; about 0.5 credits a model). Ask before a batch: every model costs money.

### Rodin from an agent, on any plan (MCP)

Hyper3D's MCP server bills the workspace you sign in with, at its plan and credits, so it works on Creator where
the REST API refuses. Add it once, for every project, and sign in (a browser window opens). Its tools load in the
next session.

```
claude mcp add --transport http hyper3d-rodin --scope user https://api.hyper3d.com/api/mcp
claude mcp login hyper3d-rodin
```

The tools an agent uses for a prop:

| Tool | What it does | Costs credits |
|---|---|---|
| `rodin_create_uploads` | Gives a one-hour URL per picture; `PUT` the file's bytes there, then pass the upload id | no |
| `rodin_generate` | Pictures or text to a textured model: `tier`, `mesh_mode` (`Raw` = triangles, `Quad`), `quality_override` (face count), `geometry_file_format` (`glb`) | yes |
| `rodin_wait`, `rodin_get_status` | Waits up to 45 s, or reads the stage (mesh, texture, pack) | no |
| `rodin_get_result` | The result page and signed download URLs (`base_basic_pbr.glb`, `base_basic_shaded.glb`) | no |
| `rodin_generate_polygen` | A low-poly remesh of a finished model | yes |
| `rodin_generate_agentic` | Beta: builds hard-surface subjects from editable n-gons | yes |
| `rodin_generate_bang`, `rodin_generate_rigging` | Split a model into parts; add a skeleton and animations | yes |
| `rodin_list_uploads`, `rodin_delete_uploads`, `rodin_list_animation_*` | The upload library and the animation catalogue | no |

There is no balance tool; the website shows the credits left. One picture with `tier: Gen-2.5-High`,
`mesh_mode: Raw`, `quality_override: 6000`, `geometry_file_format: glb` took about two minutes and gave a 6,000-
triangle model with a 7 MB PBR `.glb`, as good as the same settings on the website (Gen-2.5 on the website costs
about 0.5 credits a model). Download it outside the repository and check it with `blender_render.py` like any other.

## Make wall art

```
python art_run.py jobs/example-art.json
```

Each job has a `theme`, `name`, `prompt`, `seed` and a `frame` (`main` 896 × 1088, `square` 1024 × 1024,
`small` 832 × 1152) or its own `width` and `height`. Pictures land in `<out>/art/<theme>/raw/`.

## The scripts

| Script | Runs in | Does |
|---|---|---|
| `asset_pipeline.py` | Python | The whole chain for a list of assets, one phase at a time so the GPU loads each model once |
| `comfy_run.py` | Python | Text to image through ComfyUI's HTTP API (FLUX schnell or SDXL; `"tile": true` for seamless patterns) |
| `comfy_3d.py` | Python | Image to mesh with Hunyuan3D 2 |
| `pick_source.py` | Python | Several seeds, keeps the most symmetric picture |
| `forge.py`, `llm.py` | Python | A brief in, a checked prop out, repeatably, with a local model in Ollama planning and judging |
| `rodin.py` | Python | Pictures or text to a textured model with Hyper3D Rodin (paid, hosted; API needs a Business plan) |
| `art_run.py` | Python | Wall art |
| `blender_lowpoly.py` | Blender | Cleans the mesh, reduces it, projects the picture as its texture, normalises it |
| `blender_render.py` | Blender | A review sheet: `blender -b -P blender_render.py -- IN.glb OUT.png --views 4` |
| `ortho_views.py` | Blender | Front, side, top and back with a measuring grid |
| `blender_assemble.py` | Blender | Joins parts into one animated `.glb` from a JSON spec |
| `scan_cut.py`, `scan_clean.py` | Blender | Clean up a phone scan |
| `start-comfyui.ps1`, `stop-comfyui.ps1` | PowerShell | Start and stop ComfyUI (set `COMFYUI_DIR`) |
| `config.py` | Python | Settings |

The workflow graphs sent to ComfyUI are built in code (`workflow()` in `comfy_run.py` and `comfy_3d.py`), not
loaded from saved workflow files.

## What goes wrong

- **Thin parts** (cords, wires, slim legs) come out lumpy or vanish. Ask for thick, solid parts, or build the
  piece by hand in Blender.
- **The back is a guess.** The mesh model sees only the front, and the texture is the front picture projected
  through.
- **Pairs never match.** Mirror one model; do not generate two.
- **One GPU job at a time.** Do not start a second batch while one runs.

More on choosing between this and the other ways to get a model: the repository's `docs/models.md`.
