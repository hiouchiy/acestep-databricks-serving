# Databricks notebook source
# MAGIC %md
# MAGIC # ACE-Step 1.5 XL-SFT — register for Model Serving (Serverless GPU `GPU_1xA10` runs the smoke test; CPU serverless also works — smoke test is then skipped)
# MAGIC
# MAGIC 1. install the pinned runtime (torch 2.10/cu128 — what upstream pins for linux x86_64)
# MAGIC 2. clone ACE-Step-1.5 at a pinned SHA (source is bundled as `code_paths`, not pip-installed:
# MAGIC    its pyproject depends on a local `nano-vllm` path that pip can't resolve; we use the `pt` LM backend)
# MAGIC 3. download weights: main repo (VAE, Qwen3-Embedding text encoder, …) + XL-SFT DiT + 5Hz LM 4B
# MAGIC 4. **smoke-test on this A10** with the exact serving PyFunc (proves it fits 24 GB and measures time)
# MAGIC 5. log the PyFunc with the weights bundled as artifacts (no download at serving start →
# MAGIC    no load-timeout risk) and register to Unity Catalog with EXACT pip pins taken from this env.

# COMMAND ----------

# MAGIC %pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
# MAGIC   "torch==2.10.0" "torchvision==0.25.0" "torchaudio==2.10.0" "torchcodec==0.10.0" "torchao==0.16.0" \
# MAGIC   "transformers>=4.51.0,<4.58.0" "diffusers>=0.37.0" "accelerate>=1.12.0" "safetensors==0.7.0" \
# MAGIC   "einops>=0.8.1" "loguru>=0.7.3" "numba>=0.63.1" "vector-quantize-pytorch>=1.27.15" "scipy>=1.10.1" \
# MAGIC   "soundfile>=0.13.1" "toml" "peft>=0.18.0" "pytorch-wavelets>=1.3.0" "pywavelets>=1.9.0" "diskcache" \
# MAGIC   "xxhash" "pyyaml" "matplotlib>=3.7.5" "imageio-ffmpeg" "huggingface-hub" "hf_transfer" \
# MAGIC   "mlflow==3.12.0" "sqlalchemy>=2.0,<2.1" "databricks-sdk>=0.102.0" "protobuf>=5.29.5,<6"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import json, os, shutil, subprocess, sys, time, traceback
dbutils.widgets.text("uc_model", "<CATALOG>.<SCHEMA>.acestep_v15_xl_sft")
dbutils.widgets.text("lm_model", "acestep-5Hz-lm-1.7B")   # default LM (fits A10 next to XL DiT)
dbutils.widgets.text("pyfunc_path", "")
UC_MODEL = dbutils.widgets.get("uc_model")
LM_MODEL = dbutils.widgets.get("lm_model")
PYFUNC_SRC = dbutils.widgets.get("pyfunc_path")
ACESTEP_SHA = "ca1e85fe9430179831e6bc6be790c332190a3866"   # main HEAD 2026-08-29 (pinned)

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

base = "/tmp"   # /local_disk0 is not writable on serverless GPU; /tmp is the same ~490 GB volume
for p in ["/", "/tmp", "/local_disk0"]:
    if os.path.isdir(p):
        log(f"df {p}: " + subprocess.run(["df", "-h", p], capture_output=True, text=True).stdout.splitlines()[-1])
WORK = f"{base}/acestep"; REPO = f"{WORK}/ACE-Step-1.5"; CKPT = f"{WORK}/checkpoints"
os.makedirs(WORK, exist_ok=True)
import torch
HAS_GPU = torch.cuda.is_available()
log(f"torch {torch.__version__} cuda={HAS_GPU} gpu={torch.cuda.get_device_name(0) if HAS_GPU else None}")

# COMMAND ----------

if not os.path.isdir(REPO):
    subprocess.run(["git", "clone", "-q", "https://github.com/ace-step/ACE-Step-1.5.git", REPO], check=True)
subprocess.run(["git", "-C", REPO, "checkout", "-q", ACESTEP_SHA], check=True)
sys.path.insert(0, REPO)
log("repo @ " + subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())

# COMMAND ----------

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"
from huggingface_hub import snapshot_download
t0 = time.time()
snapshot_download("ACE-Step/Ace-Step1.5", local_dir=CKPT, max_workers=16)
snapshot_download("ACE-Step/acestep-v15-xl-sft", local_dir=f"{CKPT}/acestep-v15-xl-sft", max_workers=16)
# the 1.7B LM ships inside the main repo; also bundle the 4B so a larger GPU can switch via ACESTEP_LM_MODEL
snapshot_download("ACE-Step/acestep-5Hz-lm-4B", local_dir=f"{CKPT}/acestep-5Hz-lm-4B", max_workers=16)
shutil.rmtree(f"{CKPT}/.cache", ignore_errors=True)
for d in sorted(os.listdir(CKPT)):
    log(f"  {d}: " + subprocess.run(["du", "-sh", f"{CKPT}/{d}"], capture_output=True, text=True).stdout.split()[0])
log(f"download done in {time.time()-t0:.0f}s")

# COMMAND ----------

# MAGIC %md ## Smoke test with the exact serving PyFunc on this A10

# COMMAND ----------

shutil.copy(PYFUNC_SRC, f"{WORK}/acestep_pyfunc.py")
SMOKE = None
if not HAS_GPU:
    log("no GPU on this compute -> skipping smoke test (verify on the serving endpoint instead)")
else:
    sys.path.insert(0, WORK)
    import importlib, base64, pandas as pd
    import acestep_pyfunc
    importlib.reload(acestep_pyfunc)

    class _Ctx:  # minimal stand-in for mlflow's PythonModelContext
        artifacts = {"checkpoints": CKPT}
        model_config = {"dit_model": "acestep-v15-xl-sft", "lm_model": LM_MODEL, "lm_backend": "pt"}

    m = acestep_pyfunc.AceStepModel()
    t0 = time.time(); m.load_context(_Ctx()); log(f"load_context {time.time()-t0:.0f}s")
    print(json.dumps(json.loads(m.predict(None, pd.DataFrame([{"action": "status"}]))["result_json"][0]), indent=1)[:1500])

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    req = pd.DataFrame([{"action": "generate", "params_json": json.dumps({
        "caption": "cinematic orchestral trailer music, epic drums, choir, strings, dark to triumphant",
        "lyrics": "[Instrumental]", "duration": 30, "seed": 42, "audio_format": "mp3"})}])
    r = m.predict(None, req)
    err = r["error"][0]
    res = json.loads(r["result_json"][0])
    log(f"generate 30s: wall {time.time()-t0:.0f}s  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB  err={err!r}")
    if err:
        raise RuntimeError(err)
    a = res["audios"][0]
    log(f"audio: {a['format']} {a['sample_rate']}Hz {a['seconds']}s seed={a['seed']} b64={len(a['audio_b64'])//1024}KB  time_costs={res['time_costs']}")
    SMOKE = {"gen_wall_s": round(time.time()-t0, 1), "peak_vram_gib": round(torch.cuda.max_memory_allocated()/2**30, 2),
             "time_costs": res["time_costs"]}
    del m; import gc; gc.collect(); torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ## Log + register (weights bundled as artifacts, exact pins from this env)

# COMMAND ----------

from importlib.metadata import version, PackageNotFoundError
PKGS = ["torch", "torchvision", "torchaudio", "torchcodec", "torchao", "transformers", "diffusers", "accelerate",
        "safetensors", "einops", "loguru", "numba", "vector-quantize-pytorch", "scipy", "soundfile", "toml", "peft",
        "pytorch-wavelets", "PyWavelets", "diskcache", "xxhash", "PyYAML", "matplotlib", "imageio-ffmpeg",
        "huggingface-hub", "numpy", "pandas", "mlflow", "sqlalchemy", "tokenizers", "sentencepiece", "protobuf"]
pins = ["--extra-index-url https://download.pytorch.org/whl/cu128"]
for p in PKGS:
    try:
        pins.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        pass
log("pins: " + ", ".join(pins))

import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
sig = ModelSignature(
    inputs=Schema([ColSpec("string", "action", required=False), ColSpec("string", "params_json", required=False),
                   ColSpec("string", "src_audio_b64", required=False), ColSpec("string", "src_audio_ext", required=False),
                   ColSpec("string", "reference_audio_b64", required=False),
                   ColSpec("string", "reference_audio_ext", required=False)]),
    outputs=Schema([ColSpec("string", "result_json"), ColSpec("string", "error")]))

code_dir = f"{WORK}/code/acestep"
shutil.rmtree(f"{WORK}/code", ignore_errors=True)
# keep the whole package (handler -> lora_manager imports acestep.training); only drop tests/caches
shutil.copytree(f"{REPO}/acestep", code_dir, ignore=shutil.ignore_patterns("*_test.py", "__pycache__"))
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/acestep_register")
t0 = time.time()
with mlflow.start_run(run_name="acestep_v15_xl_sft"):
    mlflow.log_dict({"acestep_sha": ACESTEP_SHA, "lm_model": LM_MODEL, "smoke": SMOKE}, "build_info.json")
    info = mlflow.pyfunc.log_model(
        name="acestep",
        python_model=f"{WORK}/acestep_pyfunc.py",
        code_paths=[code_dir],
        artifacts={"checkpoints": CKPT},
        model_config={"dit_model": "acestep-v15-xl-sft", "lm_model": LM_MODEL, "lm_backend": "pt"},
        signature=sig,
        pip_requirements=pins,
        registered_model_name=UC_MODEL,
    )
log(f"logged+registered in {time.time()-t0:.0f}s: {info.model_uri}")
from mlflow import MlflowClient
v = max(int(x.version) for x in MlflowClient(registry_uri="databricks-uc").search_model_versions(f"name='{UC_MODEL}'"))
dbutils.notebook.exit(json.dumps({"uc_model": UC_MODEL, "version": v, "smoke": SMOKE}))
