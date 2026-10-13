# ACE-Step 1.5 XL-SFT on Databricks — generate and serve music on one A10 GPU

## What this is

Run **[ACE-Step 1.5 XL-SFT](https://huggingface.co/ACE-Step/acestep-v15-xl-sft)** on **Databricks**: a 4B DiT music
generator plus a 5 Hz LM. One API exposes every capability the checkpoint supports and accepts any upstream
parameter:

- **text2music** (vocals in many languages, or instrumental), **cover**, **repaint**, **flow-edit**, retake variations;
- **LM helpers**: `understand` (audio → caption / lyrics / BPM / key …), `create_sample` (idea → full song spec),
  `format_sample`, `audio_to_codes`;
- every field of upstream `GenerationParams` / `GenerationConfig` (~90 knobs) via `params_json`, discovered
  dynamically with `{"action": "list_params"}`.

It runs on **1× A10 (24 GB)**.

**Licence:** ACE-Step 1.5 is MIT. The model card states the training data is licensed / royalty-free / synthetic and
that **generated music can be used commercially**. It is the **default music model** of ai-video-creation-studio.

## Ways to run

| | Way | Entry point | Use it for |
|---|---|---|---|
| 1 | **Notebook on serverless GPU** | [`notebooks/generate.py`](notebooks/generate.py) | one song at a time, interactively |
| 2 | **Batch with the AI Runtime CLI** | [`air_job/generate.yaml`](air_job/generate.yaml) (`air run`) | an unattended job that writes to a Volume |
| 3 | **Model Serving** (GPU endpoint) | [`serving/`](serving/), [`scripts/`](scripts/) | a REST endpoint for apps; API in [docs/SERVING-API.md](docs/SERVING-API.md) |

The generation engine is the serving PyFunc, [`serving/acestep_pyfunc.py`](serving/acestep_pyfunc.py); its `predict`
is the whole API. Ways 1 and 2 load it in-process through [`acestep_core.py`](acestep_core.py), so they generate
exactly as the endpoint does. The core holds:
- the runtime pins;
- the pinned ACE-Step commit;
- the checkpoint list;
- `prepare` (clone + download), `load_model` and `run`.

The notebooks and the AIR job install the runtime with the same [`scripts/install.sh`](scripts/install.sh).

## Prerequisites and configuration

| Requirement | Notes |
|---|---|
| A **Databricks workspace** | **Unity Catalog**, **serverless GPU / AI Runtime** (A10) and serverless jobs, and **GPU Model Serving** (`GPU_MEDIUM` = A10). |
| **Permissions** | `CREATE MODEL` on a UC schema, write to a Volume, create serving endpoints. |
| **Databricks CLI** (`databricks`) | authenticated: `databricks auth login --host https://<workspace> --profile <PROFILE>`. |
| **AI Runtime CLI** (`air`) | Only for way 2. |

Placeholders used below — replace them with your values:

| Placeholder | Meaning | Where it is set |
|---|---|---|
| `<PROFILE>` | CLI profile | every `--profile` / `PROFILE=` |
| `<CATALOG>.<SCHEMA>` | where the model is registered and the Volume lives | `UC_MODEL=` of `scripts/register.sh` / `create_endpoint.sh` |
| `<VOLUME>` | UC Volume for generated audio: `/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>` | the notebook's `output_dir` widget; [`air_job/generate.yaml`](air_job/generate.yaml) `DEST` |
| `<ENDPOINT>` | serving endpoint name (default `acestep-v15-xl-sft`) | `ENDPOINT=` of `scripts/create_endpoint.sh` |

Layout:

```
acestep_core.py               runtime pins, pinned source, checkpoints; prepare / load_model / run (shared by 1–3)
scripts/install.sh            runtime install on serverless GPU (notebooks and the AIR job)
notebooks/generate.py         way 1
notebooks/register_serving.py way 3: pinned install → weights → (GPU smoke test) → log + register to UC
air_job/generate.yaml         way 2;  generate.py is the CLI it runs
serving/acestep_pyfunc.py     the MLflow PyFunc (models-from-code): the whole API
scripts/register.sh           sync the repo + run notebooks/register_serving as a one-off serverless job
scripts/create_endpoint.sh    create / re-point the serving endpoint
client/acestep_client.py      small Python client (handles scale-from-zero + token refresh)
tests/e2e_all_actions.py      end-to-end test of every action against a live endpoint
examples/                     a full Japanese song; an instrumental trailer cue
docs/SERVING-API.md           endpoint API (single source)
```

## 1. Notebook on serverless GPU

1. Put the repo in the workspace, so the notebook has `../acestep_core.py`, `../serving/` and `../scripts/` next to it:
   - **Git folder** (recommended): *Workspace → Create → Git folder* with this repo's URL; or
   - **sync** a local clone: `databricks sync . /Workspace/Users/<you>/model-acestep-on-databricks --profile <PROFILE>`.
2. Open `notebooks/generate` and attach it to **serverless GPU** compute: accelerator **A10**, environment version **4**.
3. Fill in the widgets and *Run all*.
   - `params_json` takes **any** `generate` parameter of the endpoint, e.g. `caption`, `lyrics`, `duration`, `bpm`,
     `keyscale`, `vocal_language`, `seed`, `audio_format`.
   - `output_dir` is a folder in your Volume.

   The first cell installs the runtime (torch 2.10/cu128) and restarts Python. The source and ~36 GB of checkpoints
   go to `work_dir` (default `/tmp/acestep`). The song plus a `.json` with the LM metadata and timings are written to
   `output_dir`, and the song plays inline.

The same notebook runs unattended as a job on serverless GPU:

```bash
databricks jobs submit --no-wait --profile <PROFILE> --json '{
  "run_name": "acestep-generate",
  "tasks": [{"task_key": "generate",
             "notebook_task": {"notebook_path": "/Workspace/Users/<you>/model-acestep-on-databricks/notebooks/generate",
                               "base_parameters": {"output_dir": "/Volumes/<CATALOG>/<SCHEMA>/<VOLUME>/acestep"}},
             "environment_key": "gpu", "compute": {"hardware_accelerator": "GPU_1xA10"}}],
  "environments": [{"environment_key": "gpu", "spec": {"client": "4"}}]}'
```

`environments[].spec.client: "4"` is required for serverless GPU.

## 2. Batch with the AI Runtime CLI

Set `DEST` in [`air_job/generate.yaml`](air_job/generate.yaml) and the `generate.py` flags in its `command`, then:

```bash
COPYFILE_DISABLE=1 air run --file air_job/generate.yaml --watch --profile <PROFILE>
```

The job:
1. runs `scripts/install.sh`;
2. runs [`generate.py`](generate.py), which prepares the source and checkpoints and sends one `generate` request;
3. copies the song (and its `.json`) to the Volume.

`generate.py --caption … --lyrics … --duration … --seed …` covers the common case. `--params '{...}'` takes any
`generate` parameter. The output extension sets `audio_format` (`mp3` / `flac` / `wav` …).

## 3. Model Serving

```bash
# 1) register (≈ 25 min: downloads ~36 GB of weights and bundles them as MLflow artifacts)
PROFILE=<PROFILE> UC_MODEL=<CATALOG>.<SCHEMA>.acestep_v15_xl_sft ./scripts/register.sh      # COMPUTE=gpu adds a smoke test
databricks jobs get-run <run_id> --profile <PROFILE>           # wait for TERMINATED/SUCCESS
databricks model-versions list <CATALOG>.<SCHEMA>.acestep_v15_xl_sft --profile <PROFILE>

# 2) endpoint (first build ≈ 1 h: container image with the bundled weights)
PROFILE=<PROFILE> UC_MODEL=<CATALOG>.<SCHEMA>.acestep_v15_xl_sft VERSION=<n> ./scripts/create_endpoint.sh
databricks serving-endpoints get <ENDPOINT> --profile <PROFILE> | jq '.state'

# 3) use it
export DATABRICKS_HOST=https://<workspace> DATABRICKS_CONFIG_PROFILE=<PROFILE>
python examples/make_song.py
python tests/e2e_all_actions.py --src-audio any_song.wav
```

`scripts/register.sh` syncs the repo to `/Workspace/Users/<you>/model-acestep-on-databricks` and runs
[`notebooks/register_serving.py`](notebooks/register_serving.py) as a one-off serverless job. `COMPUTE=cpu` is the
default and the most reliable; `gpu` adds a smoke test with the exact serving PyFunc.

**Endpoint environment variables** (no re-registration needed):

| Variable | Default | |
|---|---|---|
| `ACESTEP_LM_MODEL` | `acestep-5Hz-lm-1.7B` | `acestep-5Hz-lm-4B` is bundled too, but needs a **> 24 GB** GPU next to the XL DiT |
| `ACESTEP_DISABLE_LM` | false | DiT only (no `thinking`, no LM helper actions) |
| `ACESTEP_OFFLOAD_TO_CPU` / `ACESTEP_OFFLOAD_DIT_TO_CPU` / `ACESTEP_LM_OFFLOAD_TO_CPU` | false | CPU offload |
| `ACESTEP_QUANTIZATION` | none | e.g. `int8_weight_only` |
| `ACESTEP_LM_BACKEND` | `pt` | `vllm` (nano-vllm) is not packaged |

**API in one minute** ([`client/acestep_client.py`](client/acestep_client.py); the full contract is
[docs/SERVING-API.md](docs/SERVING-API.md)):

```python
from acestep_client import AceStep
ace = AceStep()
res = ace.generate(caption="japanese city pop, female vocal", lyrics="[verse]\n...", vocal_language="ja",
                   duration=60, bpm=104, seed=1, audio_format="mp3")
ace.save_audios(res, "out/citypop")             # res["lm_metadata"], res["time_costs"], res["applied"] too
ace.generate(task_type="cover", src_audio="in.mp3", caption="jazz piano trio", audio_cover_strength=0.6)
ace.understand(src_audio="in.mp3")
ace.list_params()
```

## Measured results

1× A10G, XL-SFT + LM 1.7B, 50 steps:

| | |
|---|---|
| 90 s song with Japanese vocals | **52 s** (LM 19 s + DiT 27 s + VAE 2 s); lyrics sung correctly (Whisper check) |
| 68 s instrumental cue | 44–45 s |
| VRAM after load | 14.2 GB (≈ 3 GB free) |
| Cold start (scale-from-zero) | **≈ 15–25 min** |

End-to-end test (`tests/e2e_all_actions.py`) against a live endpoint on 1× A10G: **12 / 12 passed**.

| Test | Result | Wall time |
|---|---|---|
| `status` | ✅ (includes the scale-from-zero wake-up) | 1460 s |
| `list_params` | ✅ | 0.4 s |
| `create_sample` (idea → caption + metadata) | ✅ | 8.8 s |
| `format_sample` (JP lyrics → enhanced caption, `language=ja`) | ✅ | 6.3 s |
| `generate` text2music, 30 s instrumental, mp3 | ✅ | 25.9 s |
| `generate` 40 s JP vocals, `batch_size=2`, flac | ✅ 2 takes | 50.2 s |
| `generate` DiT-only (`thinking=false`), heun, shift 3, 30 steps, fades, wav | ✅ | 10.6 s |
| `audio_to_codes` (20 s clip → 100 codes) | ✅ | 1.7 s |
| `understand` (audio → caption / BPM / key) | ✅ | 6.0 s |
| `generate` `cover` (`audio_cover_strength=0.6`) | ✅ | 8.7 s |
| `generate` `repaint` 6–12 s (`chunk_mask_mode=explicit`) | ✅ | 8.6 s |
| `generate` `extract` (base-only task) | ✅ rejected with a clear error | 0.3 s |

Not individually exercised yet: `flow_edit_morph`, `retake_*`, `reference_audio`, DCW / ADG knobs, `opus`/`aac`
output. They are passed straight through to upstream.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: acestep.training` at load | The handler's LoRA manager imports it. Bundle the **whole** `acestep` package; only tests and caches are dropped. |
| LM load OOM on A10, then "Insufficient free VRAM" for every request | XL DiT + **4B** LM does not fit 24 GB. Default to **1.7B** (upstream's recommendation for 20–24 GB). If the LM ever fails to load, it is freed and serving continues DiT-only. |
| `pip install` fails building `sqlalchemy 2.1.x` | Pin `mlflow==3.12.0` + `sqlalchemy<2.1` (in `acestep_core.RUNTIME_REQUIREMENTS`). |
| `PermissionError: /local_disk0` on serverless GPU | Write to `/tmp`, which is the same large volume (`prepare` defaults to `/tmp/acestep`). |
| Serverless GPU job "failed to start within 900 seconds" | Transient capacity. Registration does not need a GPU, so use `COMPUTE=cpu` (the default). |
| First request after idle → HTTP 504 / "exceeded the maximum allowed time" | Scale-from-zero (~15–25 min) is longer than the request timeout. Send `status` and retry (the client does), or disable scale-to-zero. |
| `nano-vllm` cannot be pip-installed (a local path dependency upstream) | The source is shipped as `code_paths`, not pip-installed, and the LM uses the `pt` backend. |
| mp3 / opus / aac export needs `ffmpeg` | The `imageio-ffmpeg` binary is put on `PATH` at load. |

Upstream is pinned to ACE-Step-1.5 commit `ca1e85fe9430179831e6bc6be790c332190a3866` (`acestep_core.ACESTEP_SHA`).
The registration pins every Python package to the exact version it finds installed.

## Licence

- **This repo's code:** MIT (see `LICENSE`).
- **ACE-Step 1.5 code and weights:** they belong to their authors (ACE Studio × StepFun) and are MIT-licensed upstream.
  Check the [model card](https://huggingface.co/ACE-Step/acestep-v15-xl-sft) and the
  [repo](https://github.com/ace-step/ACE-Step-1.5) for current terms before use.
