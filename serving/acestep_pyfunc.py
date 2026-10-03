"""ACE-Step 1.5 XL-SFT serving PyFunc — full-surface music generation API (MLflow models-from-code).

One endpoint exposes EVERY capability of ACE-Step 1.5 that the XL-SFT checkpoint supports, and lets the
caller pass ANY parameter of the upstream dataclasses (discovered dynamically, so new upstream knobs work
without changing this file).

Request (dataframe_records; one row = one operation):
    action               generate | understand | create_sample | format_sample | audio_to_codes |
                         list_params | status                                    (default: generate)
    params_json          JSON object of parameters for the action (see list_params). For `generate` it may
                         contain any GenerationParams field AND any GenerationConfig field.
    src_audio_b64        base64 audio file (wav/flac/mp3/ogg) → GenerationParams.src_audio
                         (cover / repaint / flow-edit / audio_to_codes / understand)
    reference_audio_b64  base64 audio file → GenerationParams.reference_audio (style/timbre reference)
    src_audio_ext / reference_audio_ext   file extension hint (default "wav")

Response row: result_json (JSON string) + error ("" on success).
    generate → {"audios":[{"audio_b64","format","sample_rate","seconds","seed","key","params"}...],
                "lm_metadata":{...}, "time_costs":{...}, "status_message", "applied":{...}}
"""
from __future__ import annotations

import base64
import dataclasses
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback

import mlflow
import pandas as pd

# The XL-SFT checkpoint supports the "standard" task set (model card + constants.TASK_TYPES_TURBO).
SUPPORTED_TASKS = ["text2music", "repaint", "cover", "cover-nofsq"]
BASE_ONLY_TASKS = ["lego", "extract", "complete"]
# Server-side defaults that differ from the upstream dataclass defaults (upstream defaults target turbo).
SFT_DEFAULTS = {"inference_steps": 50, "guidance_scale": 7.0}
CONFIG_DEFAULTS = {"batch_size": 1, "audio_format": "flac"}
AUDIO_FORMATS = ["flac", "wav", "wav32", "mp3", "opus", "aac"]


def _jsonable(v):
    """Make upstream values JSON-safe (drop tensors, convert numpy scalars)."""
    try:
        import numpy as np
        import torch
        if isinstance(v, torch.Tensor):
            return None if v.numel() > 64 else v.detach().cpu().tolist()
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return None if v.size > 64 else v.tolist()
    except Exception:
        pass
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items() if _jsonable(x) is not None or x is None}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


class AceStepModel(mlflow.pyfunc.PythonModel):
    # ------------------------------------------------------------------ load
    def load_context(self, context):
        t0 = time.time()
        cfg = dict(context.model_config or {})
        # Endpoint environment variables override model_config, so the LM size / offload / quantization can be
        # tuned per deployment without re-registering the model.
        env_map = {"ACESTEP_DIT_MODEL": "dit_model", "ACESTEP_LM_MODEL": "lm_model", "ACESTEP_LM_BACKEND": "lm_backend",
                   "ACESTEP_OFFLOAD_TO_CPU": "offload_to_cpu", "ACESTEP_OFFLOAD_DIT_TO_CPU": "offload_dit_to_cpu",
                   "ACESTEP_QUANTIZATION": "quantization", "ACESTEP_LM_OFFLOAD_TO_CPU": "lm_offload_to_cpu",
                   "ACESTEP_DISABLE_LM": "disable_lm"}
        for env, key in env_map.items():
            if os.environ.get(env, "") != "":
                v = os.environ[env]
                cfg[key] = v.lower() in ("1", "true", "yes") if key in ("offload_to_cpu", "offload_dit_to_cpu",
                                                                        "lm_offload_to_cpu", "disable_lm") else v
        os.environ.setdefault("HF_HUB_OFFLINE", "1")          # weights are bundled; never download at serve time
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._ffmpeg_on_path()

        # project_root/checkpoints/<component> is the layout ACE-Step expects.
        ckpt = context.artifacts["checkpoints"]
        self.project_root = tempfile.mkdtemp(prefix="acestep-root-")
        os.symlink(ckpt, os.path.join(self.project_root, "checkpoints"))

        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep import inference as ai
        from acestep import constants as C
        self.ai, self.C = ai, C

        self.dit = AceStepHandler()
        msg = self.dit.initialize_service(
            project_root=self.project_root,
            config_path=cfg.get("dit_model", "acestep-v15-xl-sft"),
            device="cuda",
            offload_to_cpu=bool(cfg.get("offload_to_cpu", False)),
            offload_dit_to_cpu=bool(cfg.get("offload_dit_to_cpu", False)),
            quantization=cfg.get("quantization") or None,
            compile_model=bool(cfg.get("compile_model", False)),
        )
        self.dit_init_msg = str(msg)[:2000]

        # 5Hz LM. Default 1.7B: ACE-Step's own GPU tiers recommend it for 20-24 GB cards (A10G ≈ 22.5 GB usable);
        # the 4B LM next to the XL DiT does not fit an A10 (verified: OOM at LM load on v2).
        self.llm = LLMHandler()
        self.lm_model = cfg.get("lm_model", "acestep-5Hz-lm-1.7B")
        self.lm_init_msg = "disabled"
        if not cfg.get("disable_lm"):
            try:
                lmsg = self.llm.initialize(
                    checkpoint_dir=os.path.join(self.project_root, "checkpoints"),
                    lm_model_path=self.lm_model,
                    backend=cfg.get("lm_backend", "pt"),
                    device="cuda",
                    offload_to_cpu=bool(cfg.get("lm_offload_to_cpu", False)),
                )
                self.lm_init_msg = str(lmsg)[:2000]
            except Exception as e:
                self.lm_init_msg = f"{type(e).__name__}: {e}"
        if not getattr(self.llm, "llm_initialized", False):
            # A failed LM load can leave partial weights on the GPU → drop it so the DiT still has room.
            import gc
            import torch
            self.llm = None
            gc.collect(); torch.cuda.empty_cache()
            self.llm = LLMHandler()          # uninitialized handler: generate_music skips the LM (thinking off)
        self.lm_ready = bool(getattr(self.llm, "llm_initialized", False))
        self.cfg = cfg
        self.lock = threading.Lock()         # one GPU → serialize generations
        self.load_seconds = round(time.time() - t0, 1)

    @staticmethod
    def _ffmpeg_on_path():
        """mp3/opus/aac export shells out to `ffmpeg`; serving images have none → use imageio-ffmpeg's binary."""
        if shutil.which("ffmpeg"):
            return
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            d = tempfile.mkdtemp(prefix="ffmpeg-bin-")
            os.symlink(exe, os.path.join(d, "ffmpeg"))
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        except Exception as e:  # mp3/opus/aac will error with a clear message; flac/wav still work
            print("ffmpeg unavailable:", e, file=sys.stderr)

    # ------------------------------------------------------------------ helpers
    def _fields(self, cls):
        out = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                d = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
                d = f.default_factory()                          # type: ignore[misc]
            else:
                d = None
            out[f.name] = _jsonable(d)
        return out

    @staticmethod
    def _write_audio(b64, ext, work, name):
        if not b64:
            return None
        ext = (ext or "wav").lstrip(".").lower()
        p = os.path.join(work, f"{name}.{ext}")
        with open(p, "wb") as f:
            f.write(base64.b64decode(b64))
        return p

    def _split_params(self, p):
        gp_fields = {f.name for f in dataclasses.fields(self.ai.GenerationParams)}
        gc_fields = {f.name for f in dataclasses.fields(self.ai.GenerationConfig)}
        gp = {**SFT_DEFAULTS, **{k: v for k, v in p.items() if k in gp_fields}}
        gc = {**CONFIG_DEFAULTS, **{k: v for k, v in p.items() if k in gc_fields}}
        unknown = sorted(set(p) - gp_fields - gc_fields)
        if isinstance(gp.get("timesteps"), str):
            gp["timesteps"] = [float(x) for x in gp["timesteps"].split(",") if x.strip()]
        return gp, gc, unknown

    # ------------------------------------------------------------------ actions
    def _generate(self, p, src, ref, work):
        gp, gc, unknown = self._split_params(p)
        task = gp.get("task_type", "text2music")
        if task in BASE_ONLY_TASKS:
            raise ValueError(f"task_type '{task}' needs the ACE-Step *base* checkpoint (acestep-v15-xl-base); "
                             f"this endpoint serves XL-SFT which supports {SUPPORTED_TASKS}")
        if task not in SUPPORTED_TASKS:
            raise ValueError(f"unknown task_type '{task}' (supported: {SUPPORTED_TASKS})")
        if task in ("repaint", "cover", "cover-nofsq") and not src and not gp.get("audio_codes"):
            raise ValueError(f"task_type '{task}' needs src_audio_b64 (or audio_codes)")
        if gp.get("flow_edit_morph") and not src:
            raise ValueError("flow_edit_morph needs src_audio_b64")
        if gc.get("audio_format") not in AUDIO_FORMATS:
            raise ValueError(f"audio_format must be one of {AUDIO_FORMATS}")
        if src:
            gp["src_audio"] = src
        if ref:
            gp["reference_audio"] = ref
        params = self.ai.GenerationParams(**gp)
        config = self.ai.GenerationConfig(**gc)
        out_dir = os.path.join(work, "out"); os.makedirs(out_dir, exist_ok=True)
        t0 = time.time()
        res = self.ai.generate_music(self.dit, self.llm, params, config, save_dir=out_dir)
        if not res.success:
            raise RuntimeError(res.error or res.status_message or "generation failed")
        audios = []
        for a in res.audios:
            path = a.get("path")
            if not path or not os.path.exists(path):
                raise RuntimeError(f"audio file was not written ({config.audio_format}); see status_message")
            t = a.get("tensor")
            secs = float(t.shape[-1]) / float(a.get("sample_rate", 48000)) if t is not None else None
            ap = a.get("params") or {}
            audios.append({"audio_b64": base64.b64encode(open(path, "rb").read()).decode("ascii"),
                           "format": config.audio_format, "sample_rate": a.get("sample_rate"),
                           "seconds": round(secs, 3) if secs else None, "seed": ap.get("seed"),
                           "key": a.get("key"), "params": _jsonable(ap)})
        ex = res.extra_outputs or {}
        return {"audios": audios, "lm_metadata": _jsonable(ex.get("lm_metadata")),
                "time_costs": _jsonable(ex.get("time_costs")), "status_message": res.status_message,
                "wall_seconds": round(time.time() - t0, 2),
                "applied": {"generation_params": _jsonable({k: v for k, v in gp.items()
                                                             if k not in ("src_audio", "reference_audio")}),
                            "generation_config": _jsonable(gc), "ignored_unknown_keys": unknown}}

    def _codes_from(self, p, src):
        codes = p.get("audio_codes") or ""
        if not codes and src:
            codes = self.dit.convert_src_audio_to_codes(src)
            if isinstance(codes, str) and codes.startswith("❌"):
                raise RuntimeError(codes)
        return codes

    @staticmethod
    def _lm_kwargs(p, allowed):
        return {k: p[k] for k in allowed if k in p}

    LM_COMMON = ["temperature", "top_k", "top_p", "repetition_penalty", "use_constrained_decoding",
                 "constrained_decoding_debug"]

    def _understand(self, p, src):
        codes = self._codes_from(p, src)
        r = self.ai.understand_music(self.llm, codes, **self._lm_kwargs(p, self.LM_COMMON))
        if not r.success:
            raise RuntimeError(r.error or r.status_message)
        return {**_jsonable(r.to_dict()), "audio_codes_len": codes.count("<|audio_code_")}

    def _create_sample(self, p):
        if not p.get("query"):
            raise ValueError("create_sample needs params_json.query (natural-language description)")
        r = self.ai.create_sample(self.llm, p["query"],
                                  **self._lm_kwargs(p, ["instrumental", "vocal_language"] + self.LM_COMMON))
        if not r.success:
            raise RuntimeError(r.error or r.status_message)
        return _jsonable(r.to_dict())

    def _format_sample(self, p):
        r = self.ai.format_sample(self.llm, p.get("caption", ""), p.get("lyrics", ""),
                                  **self._lm_kwargs(p, ["user_metadata"] + self.LM_COMMON))
        if not r.success:
            raise RuntimeError(r.error or r.status_message)
        return _jsonable(r.to_dict())

    def _list_params(self):
        C = self.C
        return {"actions": ["generate", "understand", "create_sample", "format_sample", "audio_to_codes",
                            "list_params", "status"],
                "generate": {"generation_params": self._fields(self.ai.GenerationParams),
                             "generation_config": self._fields(self.ai.GenerationConfig),
                             "server_defaults_override": {**SFT_DEFAULTS, **CONFIG_DEFAULTS}},
                "lm_helper_params": {"understand": ["audio_codes"] + self.LM_COMMON,
                                     "create_sample": ["query", "instrumental", "vocal_language"] + self.LM_COMMON,
                                     "format_sample": ["caption", "lyrics", "user_metadata"] + self.LM_COMMON,
                                     "audio_to_codes": ["(src_audio_b64)"]},
                "supported_task_types": SUPPORTED_TASKS,
                "base_model_only_task_types": BASE_ONLY_TASKS,
                "audio_formats": AUDIO_FORMATS,
                "valid_languages": list(getattr(C, "VALID_LANGUAGES", [])),
                "track_names": list(getattr(C, "TRACK_NAMES", []))}

    def _status(self):
        import torch
        return {"dit_model": self.cfg.get("dit_model", "acestep-v15-xl-sft"), "lm_model": self.lm_model,
                "lm_ready": self.lm_ready,
                "vram_free_gib": round(torch.cuda.mem_get_info()[0] / 2**30, 2) if torch.cuda.is_available() else None,
                "lm_backend": self.cfg.get("lm_backend", "pt"), "load_seconds": self.load_seconds,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "vram_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
                "ffmpeg": shutil.which("ffmpeg"), "dit_init": self.dit_init_msg[-500:], "lm_init": self.lm_init_msg[-500:]}

    # ------------------------------------------------------------------ predict
    def predict(self, context, model_input, params=None):
        rows = model_input.to_dict(orient="records") if isinstance(model_input, pd.DataFrame) else list(model_input)
        out = []
        for row in rows:
            work = tempfile.mkdtemp(prefix="acestep-req-")
            try:
                action = (row.get("action") or "generate").strip().lower()
                pj = row.get("params_json")
                p = json.loads(pj) if isinstance(pj, str) and pj.strip() else (pj if isinstance(pj, dict) else {})
                src = self._write_audio(row.get("src_audio_b64"), row.get("src_audio_ext"), work, "src")
                ref = self._write_audio(row.get("reference_audio_b64"), row.get("reference_audio_ext"), work, "ref")
                if action == "list_params":
                    r = self._list_params()
                elif action == "status":
                    r = self._status()
                else:
                    with self.lock:
                        if action in ("understand", "create_sample", "format_sample") and not self.lm_ready:
                            raise RuntimeError(f"action '{action}' needs the 5Hz LM, which is not loaded "
                                               f"({self.lm_init_msg[-300:]})")
                        if action == "generate":
                            r = self._generate(p, src, ref, work)
                        elif action == "understand":
                            r = self._understand(p, src)
                        elif action == "audio_to_codes":
                            if not src:
                                raise ValueError("audio_to_codes needs src_audio_b64")
                            codes = self._codes_from({}, src)
                            r = {"audio_codes": codes, "num_codes": codes.count("<|audio_code_")}
                        elif action == "create_sample":
                            r = self._create_sample(p)
                        elif action == "format_sample":
                            r = self._format_sample(p)
                        else:
                            raise ValueError(f"unknown action '{action}'")
                out.append({"result_json": json.dumps(r, ensure_ascii=False), "error": ""})
            except Exception as e:
                out.append({"result_json": "{}", "error": f"{type(e).__name__}: {e}",
                            } if not os.environ.get("ACESTEP_DEBUG") else
                           {"result_json": json.dumps({"traceback": traceback.format_exc()}), "error": str(e)})
            finally:
                shutil.rmtree(work, ignore_errors=True)
        return pd.DataFrame(out)


mlflow.models.set_model(AceStepModel())
