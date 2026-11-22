# Forge: the model pipeline

Scripts that make Popfold props and wall art with local AI models and Blender. A prompt goes in, a low-poly
textured `.glb` that follows Popfold's prop convention comes out.

> **Alpha.** These are working scripts, not a polished tool. They were written on Windows with one GPU.

```
prompt ── FLUX ──▶ picture ── rembg ──▶ cut-out ─┬─ TRELLIS.2 ──▶ textured low-poly mesh ── Blender (turn, size) ──┬─▶ .glb + review sheet
                                                └─ Hunyuan3D 2 ──▶ dense shape ── Blender (low poly, project) ────┘
```

## What you need

| Thing | Notes |
|---|---|
| ComfyUI, running | With `flux1-schnell-fp8.safetensors` (or Chroma, for `--painter chroma`: see below) and `hunyuan3d-dit-v2.safetensors` in `models/checkpoints/` (`sd_xl_base_1.0.safetensors` too, for tileable patterns). A 12 GB GPU is enough. |
| TRELLIS.2 files (optional, `forge.py`'s default mesh step) | ComfyUI 0.46 or newer, and six files (9.6 GB) from ComfyUI's own Hugging Face organisation; see "TRELLIS.2" below. |
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
python asset_pipeline.py jobs/example-props.json --from-image   # keep the pictures, redo mesh and Blender
python asset_pipeline.py jobs/example-props.json --from-mesh    # redo only the Blender step
python asset_pipeline.py jobs/example-props.json --style=refined
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

A batch of briefs is a JSON list, run one after another (`jobs/example-briefs.json` is one):

```
python forge.py --batch jobs/example-briefs.json             # each entry: name, brief, style?, shape?, tris?, seed?, mesh?, fill?
python forge.py --batch jobs/example-briefs.json --fresh     # flags after the file apply to every entry
```

Keep the briefs you ran in such a file, outside the repository if they are a site's: it is the recipe that makes
the same props again.

The script makes every decision; the model only answers questions. Each round:

| Step | Who decides | What |
|---|---|---|
| plan | model, once | Subject wording, triangle budget, things to check. Any "no …" or "without …" clause is dropped: the picture model has no negative prompt, and the picker would hold it against the picture. `--fill` keeps pale panels the cut-out punches out |
| draw | fixed seeds | `--seeds` FLUX pictures; seeds come from `--seed` and the round |
| measure | numbers | Each cut-out: one object, not cut off, 4 to 75 % of the picture, a white border (a soft floor shadow is fine), symmetry. A picture that fails is out |
| pick | model, then a rule | The model scores each picture on its own; the best that passed the measures wins (ties: symmetry, then seed) |
| mesh | the pipeline | `--mesh trellis` (default): TRELLIS.2, colours on every side. `--mesh hunyuan`: Hunyuan3D 2 and the picture projected on |
| check | numbers, then model | The mesh's front silhouette must match the picture (IoU 0.55 for Hunyuan3D, 0.45 for TRELLIS.2); then the model scores `review.png` against the brief and the style's `about`. A pass needs both, and a score of `--pass` or more |
| next | a rule | A wrong drawing: the model rewrites the subject, new seeds. A poor mesh: the next-best picture and 1.5x the triangles |

The bars are `GATES` at the top of `forge.py`.

What tuning them taught us (keep these in mind before changing a bar or a prompt):

- **A small model writes rules into a subject.** Qwen3-VL 8B wrote "no hole" into a donut and the picker then
  marked every real donut down. Prompt wording did not stop it; `positive()` drops such clauses in code.
- **A picture has no scale, and a square canvas makes things short.** "A floor lamp" came back as four table lamps,
  and so did "a tall floor-standing lamp, as tall as a person, its pole three times the height of its shade", in
  two rounds more. The plan now says the object's outline (`tall`, `wide` or `square`) and FLUX draws on a canvas of
  that shape (768 x 1344, 1344 x 768, 1024 x 1024); `--shape` overrides it.
- **A subject that drops the brief is asked again.** Once the plan gained a `shape` field, the planner answered a
  detailed walnut, cream and brass floor lamp with just "floor-standing lamp" (and called the lamp a figure). A
  subject shorter than a third of the brief is asked for again with the next seed; after three tries the brief
  itself is drawn. The same goes for colours: a dinosaur's "teal green body, cream belly" vanished from its subject
  and FLUX drew grey and yellow ones, so every colour word in the brief (`COLOURS`) must reach the subject.
- **Pick the style for the object.** `toy` makes everything chunky and rounded; mid-century chairs came out as
  toddler furniture until they were drawn in `refined`.
- **The judge and the picker must know the style.** Told to judge "a prop in a toy house", the judge failed a good
  refined chair for not being chunky, and the picker scored four good sofas 1 or 2, calling their legs thin and
  inventing "missing back cushions". Both are given the style's `about`; the judge is told small details do not
  matter at this size, the picker to look closely before calling a part missing.
- **Measure the border, not the whole background.** Furniture on white casts a soft floor shadow that the cut-out
  leaves behind; checking the whole background threw out half of the good chair pictures.
- **The silhouette bar depends on the mesh model.** See the TRELLIS.2 notes below.
- **Show a small model one picture at a time.** Given four pictures in one request, Qwen3-VL 8B saw each one small
  and miscounted: four sofas with three back cushions each were "missing" one, and all scored 1 to 3. Scored one
  by one (with the problems written before the score), the same pictures got 7, 9, 5 and 5 with correct reasons,
  including spotting the one two-seater. It costs four calls instead of one, all kept in the cache.
- **Pin the answer's shape in the schema.** The picker once answered `{"scores": []}`, which an open schema
  allowed, and every picture counted as unusable. `pick()` now asks for exactly one score per picture, numbered 1
  to n, and asks again with the next seed if it is still short.
- **The model ties and flatters.** It gave four chair pictures the same 6 despite "no ties", and passes most
  finished models with 10/10. The measures and the tie-break (symmetry, then seed) do the real choosing; read
  every `review.png` yourself.
- **The planner's triangle budget is low for round things and figures.** It asks for about 1000; a chair with legs
  holds up at that, but a curved shell looks faceted and a standing figure needs 3000. The plan now says whether
  the brief is a figure (a character, creature, animal or person), and a figure gets at least `FIGURE_TRIS` (3000).
  `--tris N` sets any budget directly. The retry rule raises it 1.5x a round.
- **The planner softens what it finds rude.** Asked for an adult novelty creature, it rewrote the parts that made
  it one as "rounded tips". `--subject "..."` draws exactly that wording; later rounds keep it and change only the seeds.
- **FLUX schnell leaves out what its training data left out.** Told outright, in three wordings, to draw adult
  novelty shapes, it drew mushroom caps. `--painter chroma` draws with Chroma1-HD instead (Apache 2.0, built on FLUX
  schnell and trained without that filtering; 26 steps, so about ten times slower). It needs, from Comfy-Org's and
  ComfyUI's own Hugging Face repositories, `Chroma1-HD-fp8mixed.safetensors` in `models/diffusion_models/`,
  `t5xxl_fp8_e4m3fn_scaled.safetensors` in `models/text_encoders/` and FLUX's `ae.safetensors` in `models/vae/`.

**Repeatable.** Every step is keyed by a hash of what goes into it: the prompt, the seed, the input files, the
scripts that do the work and the language model's digest. A step whose key matches what is on disk is skipped.
The model is asked at temperature 0 with a fixed seed, and `llm.py` keeps every answer in `<out>/llm-cache/`, so
the same question gets the same answer without loading the model. Running the same command twice gives the same
files, the second time in seconds; a run that stopped part way picks up where it stopped. `--fresh` redoes every
step and asks the model again. A different `--seed` gives a different prop.

What repeats from scratch and what does not (tested: the same brief run twice with `--fresh`, every file
compared): the plan, the scores, every FLUX picture, the measures, the cut-out and the mask came out identical to
the bit. TRELLIS.2's mesh did not: its sparse 3D layers on the GPU give a slightly different mesh each run, even
with ComfyUI started with `--deterministic` (two runs on one cut-out: 2,049,828 and 2,049,416 bytes). A small
difference can tip the judge and send a run down another path. So a run repeats exactly from its kept steps;
from scratch it repeats up to the picture and cut-out. Keep the files of a model you want (`<out>/models/` does).

It writes `<out>/props/<name>/` (`<name>` from `--name`, or the brief's first words): `r1/`, `r2/` ... with each
round's candidate pictures, cut-out, meshes and sheet, `progress/` (one labelled picture per stage, in order: the
candidates with their scores and which won, then the cut-out beside the model's sheet and verdict; it prints each
path as it writes it), the chosen `<name>.glb` and `review.png`, and `run.json`
(every input, measure, score and decision, and the versions used). Every round's model, passed or not, is also
copied to `<out>/models/<name>-r<round>-s<seed>.glb`, with TRELLIS.2's dense original beside it as `...-dense.glb`,
so nothing made is lost when a later run reuses the folder. When no round passes, the best round is still
kept and `run.json` says it did not pass. **Look at `review.png` before using a prop.**

`llm.banned` in `forge.json` is a list of words (brand or character names) that must never reach a prompt: a
brief that uses one is refused, and a plan or rewrite that uses one is asked again. `llm.py` is the small Ollama
client.

The GPU is shared: a model call that is not in the cache frees ComfyUI's models first, and Ollama unloads the
language model as soon as it answers.

### Look at every model: the viewer

```
python viewer.py                    # http://127.0.0.1:5195
python viewer.py --host 0.0.0.0     # also from a phone on the same network, or through a tunnel to the port
```

A grid of every model in `<out>`: the props (each final model, with its review sheet and progress pictures), every
round's model in `<out>/models`, the dense ones and the test meshes. Tap one to turn it around, in wireframe if
you like, with what its `run.json` says: the brief, the judge's score and problems, the front match, triangles and
size, and what made it ("Made by": the picture, mesh and clean-up models, or the hosted service). A link to one
model is the page's address with `#<file>`. An open page picks up new models every 30 seconds. It serves `<out>` read-only, with three.js from
Popfold's `node_modules` and nothing but Python's standard library. The first browser to show a model draws its
thumbnail and sends it back to `<out>/viewer-thumbs/`, so a phone downloads a small picture per model, not
every model. What comes through a tunnel cannot save thumbnails: leave `http://127.0.0.1:5195/?thumbs` open on this
machine and it draws every missing one as models arrive.

Side-by-side trials go in `<out>/compare/<way>/<prop>/<prop>.glb`, with `review.png` and a `log.json`
(`wall_seconds`, `credits`, `triangles`, `notes`, and optionally `maker` and `made_by`); they show in the Compare
tab.

To try a 3D model on Hugging Face's GPUs before installing it, and what that costs, see
[HUGGINGFACE.md](HUGGINGFACE.md).

## TRELLIS.2: colours on every side, locally

Hunyuan3D 2 makes only the shape; the colours are the front picture projected through, so sides and backs are a
blurred guess. TRELLIS.2 (Microsoft, MIT licence) makes the shape **and** its colours all round. ComfyUI 0.46 has
its nodes built in; `comfy_trellis.py` builds the graph in code, following ComfyUI's own template, and ends with
ComfyUI remeshing it into one solid, decimating it to a dense 20,000 faces, unwrapping it and baking the colours
into a texture. `blender_fit.py` then reduces it to the triangle budget, turns it to face front (by the same
silhouette test as `blender_lowpoly.py`) and sizes it. `forge.py` uses it unless you pass `--mesh hunyuan`.

```
python comfy_trellis.py picture.png --faces 3000 --texture 1024      # one picture, outside forge.py
```

The files, from `huggingface.co/Comfy-Org` (ComfyUI's repackaging of each publisher's weights, in the format its
loaders read), into ComfyUI's `models/` folder:

| File | Folder | Size | Repo, licence |
|---|---|---|---|
| `trellis_2_int8_convrot.safetensors` | `diffusion_models` | 5.25 GB | `Comfy-Org/TRELLIS.2` (from `microsoft/TRELLIS.2-4B`), MIT |
| `trellis_2_shape_vae_bf16.safetensors` | `vae` | 1.10 GB | `Comfy-Org/Pixal3D`, MIT |
| `trellis_2_texture_vae_bf16.safetensors` | `vae` | 0.95 GB | `Comfy-Org/Pixal3D`, MIT |
| `dino_v3_L_naf_fp32.safetensors` | `clip_vision` | 1.22 GB | `Comfy-Org/Pixal3D`; built on Meta's DINOv3, which has its own licence: read it before selling what you make |
| `birefnet.safetensors` | `background_removal` | 0.44 GB | `Comfy-Org/BiRefNet`, MIT |
| `moge_2_vitl_normal_fp16.safetensors` | `geometry_estimation` | 0.66 GB | `Comfy-Org/MoGe`, MIT (only for the Pixal3D half of the template; not used here) |

Check each download's SHA-256 against the `lfs.oid` Hugging Face's API gives for the file
(`/api/models/<repo>/paths-info/main`). The int8 model is the one that fits a 12 GB card; the bf16 one is 10.3 GB.

What we found on a 12 GB card (RTX 4070 SUPER):

- **It is much better.** The same donut picture gave a donut with icing, dough and sprinkles coloured on every
  side, where Hunyuan3D's had a pink smear behind. Chunky objects come out as clean shapes.
- **It is slow, and the upsample resolution decides how slow.** The template's `target_resolution` 1536 took about
  45 s a step (8 minutes for 12 steps) and the colour decode then filled all 12 GB; the whole job took 25 minutes.
  `comfy_trellis.py` uses 1024, which is plenty for a prop decimated to a few thousand faces.
- **Fill the triangles when you compare silhouettes.** The first `blender_fit.py` sampled only vertices and face
  centres: a good 1000-triangle sideboard scored 0.17 because its big flat faces left the silhouette empty. It now
  fills every triangle, and good models score 0.58 to 0.97.
- **Thin radiating parts still fail.** A sunburst wall clock's brass rays came back spiky and melted into the face
  in three rounds, from 3000 to 6000 triangles. Chairs, tables, cabinets, a sofa, a planter and a standing figure
  all worked; for rays, spokes, wires and fine frames, build the part in Blender (`blender_assemble.py`).
- **Pale parts on white get cut away.** rembg took a table lamp's cream shade for the white background and kept only
  the coral base; the model came back a ball, and its silhouette matched the (wrong) cut-out at 0.96, so no measure
  caught it. `keep_pale_parts()` in `asset_pipeline.py` now adds back any region outside the cut-out that has some
  colour (not the grey of a floor shadow), touches the object and is at least 3 % of its size.
- **Some objects are too big to decode finely on 12 GB.** A stack of folded towels built a sparse structure whose fine
  shape decode asked for 14.3 GB; once it aborted ComfyUI outright, once it failed cleanly. On out of memory,
  `comfy_trellis.run()` frees ComfyUI's models and tries once more without the finer pass (shape and colours from the
  first shape stage). `run.json` records which path a model took (`"fine": false`). This only catches a clean
  out-of-memory error: twice the towels aborted ComfyUI's process instead (`Fatal Python error: Aborted` in
  `flexgemm.py`), which no retry can catch. Restart ComfyUI (`start-comfyui.ps1`) and leave such a brief out, or
  describe a smaller, simpler object.
- **Cut the picture once.** TRELLIS.2's graph runs its own background removal (BiRefNet), which cut a round mirror's
  pale glass away after our cut-out had kept it: the model was a bare ring (the silhouette gate caught it at 0.35;
  the judge gave it 10/10). `forge.py` now sends our cut-out as the picture's alpha
  (`comfy_trellis.run(..., mask=...)`) and the graph uses that instead.
- **Chrome reads as see-through or white.** A kitchen stool's thick chrome legs came back transparent, then white
  and rod-thin, and its footrest ring merged into them, three rounds running. Mirror finishes reflect the white
  background, so the cut-out and the model treat them like glass. Ask for painted metal (a colour) instead.
- **It decides how an object rests.** A donut drawn face-on comes back lying flat, as a real one would. The front
  silhouette then matches the picture less well (about 0.63) though the model is right, so the IoU gate is a
  sanity check, not a measure of quality.
- **At 1024 a model takes about two minutes** (peak 11.4 GB), against 25 at the template's 1536.
- **Do not let ComfyUI decimate to the budget.** Its `DecimateMesh` (midpoint) is quick and crude: a chair cut to
  1000 faces came back crumpled. `forge.py` asks for 20,000 faces, ComfyUI bakes the colours onto those, and
  `blender_fit.py --tris` reduces it with Blender's collapse decimation, which keeps the UVs, so the texture still
  fits. A donut at 1000 triangles stays clean that way.
- **Its colours come out darker, and greens weaker.** Teal upholstery came back navy and mustard came back orange;
  a saturated orange came through about right. It is not one fixed shift, so a single correction does not fix it.
  It is not the crop background either: the template's black gave teal (22, 58, 87) for a picture's (38, 95, 104),
  and white (`comfy_trellis.run(..., background="#FFFFFF")`) gave a much darker (7, 29, 44). Nor is it a linear
  versus sRGB mix-up: converting the texture overshoots far too bright. Keep black, and compare `review.png` with the
  picture's colours before using a model.
- **The output is already textured**, so `blender_lowpoly.py` (which re-projects the picture) is not used;
  `blender_fit.py` keeps TRELLIS.2's texture and writes it as JPEG.

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
| `comfy_run.py` | Python | Text to image through ComfyUI's HTTP API (FLUX schnell, Chroma or SDXL; `"tile": true` for seamless patterns) |
| `comfy_3d.py` | Python | Image to mesh with Hunyuan3D 2 (shape only) |
| `comfy_trellis.py` | Python | Image to textured low-poly mesh with TRELLIS.2 |
| `pick_source.py` | Python | Several seeds, keeps the most symmetric picture |
| `forge.py`, `llm.py` | Python | A brief in, a checked prop out, repeatably, with a local model in Ollama planning and judging |
| `viewer.py` | Python | A web page of every model made, to turn each one around, from a desktop or a phone |
| `rodin.py` | Python | Pictures or text to a textured model with Hyper3D Rodin (paid, hosted; API needs a Business plan) |
| `art_run.py` | Python | Wall art |
| `blender_lowpoly.py` | Blender | Cleans the mesh, reduces it, projects the picture as its texture, normalises it |
| `blender_fit.py` | Blender | Turns a textured model to face front and sizes it, keeping its texture (TRELLIS.2) |
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
