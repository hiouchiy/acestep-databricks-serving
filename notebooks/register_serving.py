# Databricks notebook source
# MAGIC %md
# MAGIC # ACE-Step 1.5 XL-SFT — register for Model Serving
# MAGIC
# MAGIC Serverless GPU (`GPU_1xA10`) runs a smoke test first; CPU serverless also works, and the smoke test is then
# MAGIC skipped. Open it from a Git folder (or a synced copy) of this repo. `scripts/register.sh` uploads and runs it.
# MAGIC
# MAGIC 1. Install the pinned runtime (`scripts/install.sh`; the pins are in `acestep_core.py`).
# MAGIC 2. Prepare the pinned ACE-Step source and the checkpoints (`acestep_core.prepare`). The source is bundled as
# MAGIC    `code_paths`, not pip-installed: its pyproject depends on a local `nano-vllm` path pip cannot resolve, and the
# MAGIC    `pt` LM backend is used instead.
# MAGIC 3. **Smoke-test on the A10** with the exact serving PyFunc. This proves it fits 24 GB and measures the time.
# MAGIC 4. Log the PyFunc and register it to Unity Catalog.
# MAGIC    - The weights are bundled as artifacts, so there is no download at serving start and no load-timeout risk.
# MAGIC    - The pip pins are the EXACT versions installed in this environment.

# COMMAND ----------

import os
import subprocess
import sys

REPO = os.path.abspath("..")
subprocess.run(["bash", os.path.join(REPO, "scripts", "install.sh")], check=True,
               env={**os.environ, "PYTHON": sys.executable})
dbutils.library.restartPython()

# COMMAND ----------

import gc
import json
import os
import shutil
import sys
import time

REPO = os.path.abspath("..")
sys.path.insert(0, REPO)
import acestep_core as core  # noqa: E402

dbutils.widgets.text("uc_model", "<CATALOG>.<SCHEMA>.acestep_v15_xl_sft")
dbutils.widgets.text("lm_model", core.LM_MODEL)   # default LM (fits A10 next to XL DiT)
UC_MODEL = dbutils.widgets.get("uc_model")
LM_MODEL = dbutils.widgets.get("lm_model")
assert "<" not in UC_MODEL, "set uc_model to <catalog>.<schema>.<model>"

WORK = "/tmp/acestep"   # /local_disk0 is not writable on serverless GPU; /tmp is the same ~490 GB volume
for p in ["/", "/tmp", "/local_disk0"]:
    if os.path.isdir(p):
        core.log(f"df {p}: " + subprocess.run(["df", "-h", p], capture_output=True, text=True).stdout.splitlines()[-1])
import torch  # noqa: E402
HAS_GPU = torch.cuda.is_available()
core.log(f"torch {torch.__version__} cuda={HAS_GPU} gpu={torch.cuda.get_device_name(0) if HAS_GPU else None}")
ACE_REPO, CKPT = core.prepare(WORK)

# COMMAND ----------

# MAGIC %md ## Smoke test with the exact serving PyFunc on this A10

# COMMAND ----------

SMOKE = None
if not HAS_GPU:
    core.log("no GPU on this compute -> skipping smoke test (verify on the serving endpoint instead)")
else:
    m = core.load_model(CKPT, lm_model=LM_MODEL)
    print(json.dumps(core.run(m, "status"), indent=1)[:1500])
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res = core.run(m, "generate", {"caption": "cinematic orchestral trailer music, epic drums, choir, strings, dark to triumphant",
                                   "lyrics": "[Instrumental]", "duration": 30, "seed": 42, "audio_format": "mp3"})
    a = res["audios"][0]
    core.log(f"generate 30s: wall {time.time()-t0:.0f}s  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
    core.log(f"audio: {a['format']} {a['sample_rate']}Hz {a['seconds']}s seed={a['seed']} b64={len(a['audio_b64'])//1024}KB  time_costs={res['time_costs']}")
    SMOKE = {"gen_wall_s": round(time.time()-t0, 1), "peak_vram_gib": round(torch.cuda.max_memory_allocated()/2**30, 2),
             "time_costs": res["time_costs"]}
    del m; gc.collect(); torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md ## Log + register (weights bundled as artifacts, exact pins from this env)

# COMMAND ----------

from importlib.metadata import PackageNotFoundError, version  # noqa: E402

pins = [f"--extra-index-url {core.TORCH_INDEX}"]
for p in core.REGISTER_PIN_PACKAGES:
    try:
        pins.append(f"{p}=={version(p)}")
    except PackageNotFoundError:
        pass
core.log("pins: " + ", ".join(pins))

import mlflow  # noqa: E402
from mlflow.models.signature import ModelSignature  # noqa: E402
from mlflow.types.schema import ColSpec, Schema  # noqa: E402
sig = ModelSignature(
    inputs=Schema([ColSpec("string", "action", required=False), ColSpec("string", "params_json", required=False),
                   ColSpec("string", "src_audio_b64", required=False), ColSpec("string", "src_audio_ext", required=False),
                   ColSpec("string", "reference_audio_b64", required=False),
                   ColSpec("string", "reference_audio_ext", required=False)]),
    outputs=Schema([ColSpec("string", "result_json"), ColSpec("string", "error")]))

code_dir = f"{WORK}/code/acestep"
shutil.rmtree(f"{WORK}/code", ignore_errors=True)
# keep the whole package (handler -> lora_manager imports acestep.training); only drop tests/caches
shutil.copytree(f"{ACE_REPO}/acestep", code_dir, ignore=shutil.ignore_patterns("*_test.py", "__pycache__"))
pyfunc = f"{WORK}/acestep_pyfunc.py"
shutil.copy(core.PYFUNC, pyfunc)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{spark.sql('select current_user()').first()[0]}/acestep_register")
t0 = time.time()
with mlflow.start_run(run_name="acestep_v15_xl_sft"):
    mlflow.log_dict({"acestep_sha": core.ACESTEP_SHA, "lm_model": LM_MODEL, "smoke": SMOKE}, "build_info.json")
    info = mlflow.pyfunc.log_model(
        name="acestep",
        python_model=pyfunc,
        code_paths=[code_dir],
        artifacts={"checkpoints": str(CKPT)},
        model_config={"dit_model": core.DIT_MODEL, "lm_model": LM_MODEL, "lm_backend": "pt"},
        signature=sig,
        pip_requirements=pins,
        registered_model_name=UC_MODEL,
    )
core.log(f"logged+registered in {time.time()-t0:.0f}s: {info.model_uri}")
from mlflow import MlflowClient  # noqa: E402
v = max(int(x.version) for x in MlflowClient(registry_uri="databricks-uc").search_model_versions(f"name='{UC_MODEL}'"))
dbutils.notebook.exit(json.dumps({"uc_model": UC_MODEL, "version": v, "smoke": SMOKE}))
