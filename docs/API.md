# API reference — ACE-Step 1.5 XL-SFT on Databricks Model Serving

```
POST https://<your-workspace>.cloud.databricks.com/serving-endpoints/<endpoint>/invocations
Authorization: Bearer <token>
Content-Type: application/json
{"dataframe_records": [ { ...one request row... } ]}
```

Each row is one operation. Send **one row per request**: the endpoint has one GPU, so rows run one after another.
The response is `{"predictions": [{"result_json": "<JSON string>", "error": ""}]}`.

**Errors:** when an operation fails (bad input, generation failure), the HTTP status is still **200**, with the
reason in `error` (e.g. `ValueError: ...`). HTTP 4xx/5xx only happen for platform-level problems: auth, timeout,
or the endpoint is still scaling from zero.

## 1. Request columns

All columns are optional strings.

| Column | Meaning |
|---|---|
| `action` | `generate` (default) / `understand` / `create_sample` / `format_sample` / `audio_to_codes` / `list_params` / `status` |
| `params_json` | A JSON object of parameters for the action (§3). For `generate`, **any field of the upstream `GenerationParams` or `GenerationConfig` dataclass** is accepted. |
| `src_audio_b64` | A base64-encoded audio file (wav / flac / mp3 / ogg). Becomes `GenerationParams.src_audio` (cover / repaint / flow-edit / understand / audio_to_codes). |
| `src_audio_ext` | Its extension (default `wav`) |
| `reference_audio_b64` | A base64-encoded style/timbre reference audio file → `GenerationParams.reference_audio` |
| `reference_audio_ext` | Its extension (default `wav`) |

The parameter surface is discovered dynamically with `dataclasses.fields()`, so knobs added upstream later work
without changing the server. Unknown keys are ignored and listed in `applied.ignored_unknown_keys`.

## 2. Actions

| action | What it does | Needs the LM |
|---|---|---|
| `generate` | Music generation (§3.1). `task_type` = `text2music` / `cover` / `cover-nofsq` / `repaint` | Optional (`thinking`) |
| `understand` | Analyzes audio (or `audio_codes`) and estimates caption / lyrics / BPM / key / time signature / language / duration | Yes |
| `create_sample` | From a natural-language description, creates a complete set of caption, lyrics and metadata (Simple / Inspiration mode) | Yes |
| `format_sample` | Polishes the user's caption and lyrics, and fills in metadata | Yes |
| `audio_to_codes` | Converts audio into 5 Hz semantic codes (`<\|audio_code_N\|>…`) for later use as `audio_codes` | No |
| `list_params` | Returns every parameter name, its default, the supported task_types, languages, formats and track names | No |
| `status` | Loaded models, GPU, VRAM, LM state, load time. **Use it to wake the endpoint from scale-to-zero.** | No |

`lego` / `extract` / `complete` need the **base checkpoint** (`acestep-v15-xl-base`). This endpoint serves XL-SFT, so
they are rejected with a clear `error`.

## 3. Parameters (`params_json`)

Server-side defaults that differ from upstream (upstream defaults target the turbo model):
`inference_steps=50`, `guidance_scale=7.0` (SFT runs CFG with 50 steps), `batch_size=1`, `audio_format="flac"`.
Get the current full list with `{"action":"list_params"}`.

### 3.1 `generate` — GenerationParams (main fields)

**Text, metadata**

| Key | Default | Description |
|---|---|---|
| `caption` | "" | Description of the music (genre, instruments, mood…), < 512 characters |
| `lyrics` | "" | Lyrics. Section tags such as `[verse]` `[chorus]`; `[Instrumental]` for an instrumental. < 4096 characters |
| `instrumental` | false | Force an instrumental |
| `vocal_language` | "unknown" | `ja` / `en` / `zh` … (`list_params.valid_languages`) |
| `bpm` | null (auto) | 30–300 |
| `keyscale` | "" (auto) | e.g. `"C Major"` `"A minor"` |
| `timesignature` | "" (auto) | `"2"` `"3"` `"4"` `"6"` |
| `duration` | -1 (auto) | Length in seconds, 10–600 |
| `global_caption` | "" | Song-level caption (for stems) |

**Sampling**

| Key | Default | Description |
|---|---|---|
| `seed` | -1 (random) | Reproducibility |
| `inference_steps` | **50** | Diffusion steps |
| `guidance_scale` | 7.0 | CFG strength |
| `shift` | 1.0 | Timestep shift |
| `timesteps` | null | Custom schedule; a list or `"0.97,0.76,…,0"` (overrides steps/shift) |
| `infer_method` | "ode" | `ode` / `sde` |
| `sampler_mode` | "euler" | `euler` / `heun` |
| `use_adg` | false | Adaptive Dual Guidance (base only) |
| `cfg_interval_start` / `cfg_interval_end` | 0.0 / 1.0 | Range over which CFG is applied |
| `velocity_norm_threshold` / `velocity_ema_factor` | 0 / 0 | Stabilization |
| `dcw_enabled` / `dcw_mode` / `dcw_scaler` / `dcw_high_scaler` / `dcw_wavelet` | auto / "double" / 0.05 / 0.02 / "haar" | DCW (wavelet-domain correction) |

**Tasks (audio-to-audio)**

| Key | Default | Description |
|---|---|---|
| `task_type` | "text2music" | `text2music` / `cover` / `cover-nofsq` / `repaint` |
| `audio_codes` | "" | Generate from semantic codes (output of `audio_to_codes`) |
| `audio_cover_strength` | 1.0 | Influence of the source in a cover (0–1; around 0.2 suits style transfer) |
| `cover_noise_strength` | 0.0 | 0 = pure noise … 1 = closest to the source |
| `repainting_start` / `repainting_end` | 0 / -1 | Repaint range in seconds (-1 = to the end) |
| `chunk_mask_mode` | "auto" | `explicit` = use the range as the mask exactly (recommended for repaint) |
| `repaint_mode` / `repaint_strength` | "balanced" / 0.5 | `conservative` / `balanced` / `aggressive` |
| `repaint_latent_crossfade_frames` / `repaint_wav_crossfade_sec` | 10 / 0.0 | Blending at the boundaries |
| `retake_seed` / `retake_variance` | null / 0.0 | Variations (variance-preserving noise mixing) |
| `flow_edit_morph` + `flow_edit_source_caption` / `flow_edit_source_lyrics` / `flow_edit_n_min` / `flow_edit_n_max` / `flow_edit_n_avg` | false … | Flow-edit: morphs the source audio toward the target caption/lyrics (on top of cover) |
| `instruction` | (auto) | Task instruction text |

**5 Hz LM (Chain-of-Thought)**

| Key | Default | Description |
|---|---|---|
| `thinking` | true | Let the LM generate metadata and codes. With false the DiT runs alone (faster) |
| `lm_temperature` / `lm_cfg_scale` / `lm_top_k` / `lm_top_p` / `lm_negative_prompt` | 0.85 / 2.0 / 0 / 0.9 / "NO USER INPUT" | LM sampling |
| `use_cot_metas` / `use_cot_caption` / `use_cot_language` / `use_constrained_decoding` | true | What the LM is allowed to fill in |
| `cot_bpm` / `cot_keyscale` / `cot_timesignature` / `cot_duration` / `cot_vocal_language` / `cot_caption` / `cot_lyrics` | — | Pin CoT values explicitly |

**Post-processing**

| Key | Default | Description |
|---|---|---|
| `enable_normalization` / `normalization_db` | true / -1.0 | Peak normalization |
| `fade_in_duration` / `fade_out_duration` | 0 / 0 | Fades in seconds |
| `latent_shift` / `latent_rescale` | 0 / 1.0 | Latent adjustment before the VAE |

### 3.2 `generate` — GenerationConfig

| Key | Default | Description |
|---|---|---|
| `batch_size` | **1** | Number of variations per request (VRAM grows with it) |
| `seeds` | null | A list of seeds, one per batch item |
| `use_random_seed` | true | |
| `audio_format` | **"flac"** | `flac` / `wav` / `wav32` / `mp3` / `opus` / `aac` |
| `mp3_bitrate` / `mp3_sample_rate` | "128k" / 48000 | |
| `allow_lm_batch` / `lm_batch_chunk_size` / `constrained_decoding_debug` | false / 8 / false | |

### 3.3 Helper actions

| action | Keys |
|---|---|
| `understand` | `audio_codes` (or `src_audio_b64`), `temperature`, `top_k`, `top_p`, `repetition_penalty`, `use_constrained_decoding` |
| `create_sample` | **`query`** (required), `instrumental`, `vocal_language`, + the sampling keys above |
| `format_sample` | `caption`, `lyrics`, `user_metadata` (dict), + the sampling keys above |

## 4. Response (`result_json`)

`generate`:
```json
{"audios": [{"audio_b64": "...", "format": "mp3", "sample_rate": 48000, "seconds": 90.0, "seed": 2026,
             "key": "...", "params": {...}}],
 "lm_metadata": {"bpm": 128, "keyscale": "D major", "timesignature": "4", "duration": 90, "caption": "..."},
 "time_costs": {"lm_total_time": 19.2, "dit_total_time_cost": 29.0, "dit_vae_decode_time_cost": 2.2, "...": "..."},
 "status_message": "...", "wall_seconds": 48.2,
 "applied": {"generation_params": {...}, "generation_config": {...}, "ignored_unknown_keys": []}}
```
`understand` / `create_sample` / `format_sample`: `caption`, `lyrics`, `bpm`, `duration`, `keyscale`, `language`,
`timesignature` (`create_sample` also returns `instrumental`). `audio_to_codes`: `audio_codes`, `num_codes`.

## 5. Examples

```bash
# Wake from scale-to-zero / check state
curl -s -X POST "$HOST/serving-endpoints/acestep-v15-xl-sft/invocations" -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" -d '{"dataframe_records":[{"action":"status"}]}'
```
```python
from acestep_client import AceStep          # client/acestep_client.py
ace = AceStep()
res = ace.generate(caption="lofi hip hop, mellow piano", lyrics="[Instrumental]", duration=30, audio_format="mp3")
ace.save_audios(res, "out/lofi")
ace.generate(task_type="cover", src_audio="song.mp3", caption="jazz piano trio", audio_cover_strength=0.6)
ace.generate(task_type="repaint", src_audio="song.mp3", repainting_start=6, repainting_end=12,
             chunk_mask_mode="explicit", caption="quiet cello breakdown")
ace.understand(src_audio="song.mp3")
ace.create_sample(query="a melancholic city-pop song about the last train", vocal_language="ja")
```

## 6. Performance and limits (measured on 1× A10G, XL-SFT + LM 1.7B, 50 steps)

| Case | Generation time |
|---|---|
| 90 s song with vocals (with LM) | **52 s** (LM 19 s + DiT 27 s + VAE 2 s) |
| 68 s instrumental (with LM) | 44–45 s |

- VRAM: about 14.2 GB in use after load (3 GB free). Larger `batch_size` and longer `duration` use more.
- The XL DiT next to the 4B LM does not fit an A10 (the LM load runs out of memory). The 4B is bundled anyway, so on a
  >24 GB GPU you can switch to it with the endpoint environment variable `ACESTEP_LM_MODEL=acestep-5Hz-lm-4B`.
- **Cold start (scale-from-zero) takes about 15–25 min**, longer than the request timeout (~10 min), so the first
  request after idle fails. Send `status` first and retry until it answers (`client/acestep_client.py` does this
  automatically), or turn scale-to-zero off while you're working.
- A response carries the audio inline as base64. Long, high-bitrate output (e.g. 10 minutes of wav) makes for a very
  large response, so use `mp3` / `opus` for long pieces.
