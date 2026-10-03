# ACE-Step 1.5 XL-SFT on Databricks Model Serving

Serve **[ACE-Step 1.5 XL-SFT](https://huggingface.co/ACE-Step/acestep-v15-xl-sft)** (4B DiT music generation + 5 Hz
LM) as a **GPU Model Serving endpoint on Databricks**, with **one API that exposes every capability the checkpoint
supports** and lets callers pass **any upstream parameter**.

- **text2music** (vocals in many languages or instrumental), **cover**, **repaint**, **flow-edit**, retake variations
- **LM helpers**: `understand` (audio → caption/lyrics/BPM/key…), `create_sample` (idea → full song spec),
  `format_sample`, `audio_to_codes`
- every field of upstream `GenerationParams` / `GenerationConfig` (~90 knobs) accepted via `params_json`,
  discovered dynamically → `{"action":"list_params"}`
- runs on **1× A10 (24 GB, `GPU_MEDIUM`)** with scale-to-zero; weights bundled in the model (no download at start)

> ACE-Step 1.5 is MIT-licensed and the model card states the training data is licensed / royalty-free / synthetic and
> **generated music can be used commercially** — see the upstream model card and repo for the authoritative terms.

## Layout

```
serving/acestep_pyfunc.py     MLflow PyFunc (models-from-code): the whole API
serving/register_acestep.py   Databricks notebook: pinned install → weights → (GPU smoke test) → log + register to UC
scripts/register.sh           upload + run the notebook as a one-off serverless job
scripts/create_endpoint.sh    create / re-point the serving endpoint
client/acestep_client.py      small Python client (handles scale-from-zero + token refresh)
tests/e2e_all_actions.py      end-to-end test of every action
examples/                     a full Japanese song; an instrumental trailer cue
docs/API.md                   full API reference (Japanese)
```

## Prerequisites

- A Databricks workspace with **Unity Catalog**, **Serverless jobs** and **GPU Model Serving** (`GPU_MEDIUM` = A10).
- Databricks CLI ≥ 0.2xx authenticated: `databricks auth login --host https://<your-workspace>.cloud.databricks.com --profile <profile>`
- `CREATE MODEL` on a UC schema: `<CATALOG>.<SCHEMA>`.

## Deploy

```bash
# 1) register (≈ 25 min: downloads ~36 GB of weights and bundles them as MLflow artifacts)
PROFILE=<profile> UC_MODEL=<CATALOG>.<SCHEMA>.acestep_v15_xl_sft ./scripts/register.sh          # COMPUTE=gpu adds a smoke test
databricks jobs get-run <run_id> --profile <profile>           # wait for TERMINATED/SUCCESS
databricks model-versions list <CATALOG>.<SCHEMA>.acestep_v15_xl_sft --profile <profile>

# 2) endpoint (first build ≈ 1 h: container image with the bundled weights)
PROFILE=<profile> UC_MODEL=<CATALOG>.<SCHEMA>.acestep_v15_xl_sft VERSION=<n> ./scripts/create_endpoint.sh
databricks serving-endpoints get acestep-v15-xl-sft --profile <profile> | jq '.state'

# 3) use it
export DATABRICKS_HOST=https://<your-workspace>.cloud.databricks.com DATABRICKS_CONFIG_PROFILE=<profile>
python examples/make_song.py
python tests/e2e_all_actions.py --src-audio any_song.wav
```

### Runtime configuration (endpoint environment variables — no re-registration needed)

| Variable | Default | |
|---|---|---|
| `ACESTEP_LM_MODEL` | `acestep-5Hz-lm-1.7B` | `acestep-5Hz-lm-4B` is bundled too, but needs a **> 24 GB** GPU next to the XL DiT |
| `ACESTEP_DISABLE_LM` | false | DiT only (no `thinking`, no LM helper actions) |
| `ACESTEP_OFFLOAD_TO_CPU` / `ACESTEP_OFFLOAD_DIT_TO_CPU` / `ACESTEP_LM_OFFLOAD_TO_CPU` | false | CPU offload |
| `ACESTEP_QUANTIZATION` | none | e.g. `int8_weight_only` |
| `ACESTEP_LM_BACKEND` | `pt` | `vllm` (nano-vllm) is not packaged |

## API in one minute

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

Request = one row `{action, params_json, src_audio_b64, src_audio_ext, reference_audio_b64, reference_audio_ext}`;
response = `{result_json, error}`. Details: **[docs/API.md](docs/API.md)**.

## Measured (1× A10G, XL-SFT + LM 1.7B, 50 steps)

| | |
|---|---|
| 90 s song with Japanese vocals | **52 s** (LM 19 s + DiT 27 s + VAE 2 s); lyrics sung correctly (Whisper check) |
| 68 s instrumental cue | 44–45 s |
| VRAM after load | 14.2 GB (≈ 3 GB free) |
| cold start (scale-from-zero) | **≈ 15–25 min** |

## Verification status

End-to-end test (`tests/e2e_all_actions.py`) against a live endpoint on 1× A10G — **12 / 12 passed**:

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
output — they are passed straight through to upstream.

## Things we hit (and how this repo handles them)

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: acestep.training` at load | the handler's LoRA manager imports it — bundle the **whole** `acestep` package (only tests/caches are dropped) |
| LM load OOM on A10, then "Insufficient free VRAM" for every request | XL DiT + **4B** LM doesn't fit 24 GB; default to **1.7B** (upstream's recommendation for 20–24 GB) and, if the LM ever fails to load, free it and keep serving DiT-only |
| `pip install` fails building `sqlalchemy 2.1.x` | pin `mlflow==3.12.0` + `sqlalchemy<2.1` (also pinned for serving) |
| `PermissionError: /local_disk0` on serverless GPU | write to `/tmp` (same large volume) |
| serverless GPU job "failed to start within 900 seconds" | transient capacity; registration doesn't need a GPU → `COMPUTE=cpu` (default) |
| first request after idle → HTTP 504 / "exceeded the maximum allowed time" | scale-from-zero (~15–25 min) is longer than the request timeout; send `status` and retry (the client does) or disable scale-to-zero |
| `nano-vllm` can't be pip-installed (local path dependency upstream) | source is shipped as `code_paths` (not pip-installed) and the LM uses the `pt` backend |
| mp3 / opus / aac export needs `ffmpeg` | `imageio-ffmpeg` binary is put on `PATH` at load (the serving image already had `/usr/bin/ffmpeg` in our runs) |

Upstream is pinned to ACE-Step-1.5 commit `ca1e85fe9430179831e6bc6be790c332190a3866`; all Python packages are pinned
to the exact versions resolved at registration time.

## License

Code in this repo: MIT (see `LICENSE`). ACE-Step 1.5 code and weights belong to their authors (ACE Studio × StepFun),
MIT-licensed upstream — check the [model card](https://huggingface.co/ACE-Step/acestep-v15-xl-sft) and
[repo](https://github.com/ace-step/ACE-Step-1.5) for current terms before use.
