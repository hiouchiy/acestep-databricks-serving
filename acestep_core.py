"""ACE-Step 1.5 XL-SFT core, shared by the notebooks, the AI Runtime batch job and the serving registration.

One place for: the runtime pins, the pinned ACE-Step source commit, the checkpoints to download, preparing a
working directory (clone + weights), and running a request. The generation engine itself is the serving PyFunc,
serving/acestep_pyfunc.py (its `predict` is the whole API, docs/SERVING-API.md), so the notebook and the batch job
generate exactly as the endpoint does.

Importing this module needs neither torch nor ACE-Step.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
PYFUNC = HERE / "serving" / "acestep_pyfunc.py"

ACESTEP_REPO = "https://github.com/ace-step/ACE-Step-1.5.git"
ACESTEP_SHA = "ca1e85fe9430179831e6bc6be790c332190a3866"   # main HEAD 2026-08-29 (pinned)
DIT_MODEL = "acestep-v15-xl-sft"
LM_MODEL = "acestep-5Hz-lm-1.7B"   # fits an A10 next to the XL DiT; the 4B LM needs a > 24 GB GPU
# (Hub repo, sub-folder of the checkpoints dir; "" = the root). The main repo carries the VAE, the Qwen3 text encoder
# and the 1.7B LM; the 4B LM is bundled too so a larger GPU can switch via ACESTEP_LM_MODEL.
WEIGHTS = [("ACE-Step/Ace-Step1.5", ""), ("ACE-Step/acestep-v15-xl-sft", "acestep-v15-xl-sft"),
           ("ACE-Step/acestep-5Hz-lm-4B", "acestep-5Hz-lm-4B")]

# The runtime (torch 2.10 / cu128 is what upstream pins for linux x86_64). scripts/install.sh installs it for the
# notebooks and the AIR job; the registration pins the exact versions it finds installed (REGISTER_PIN_PACKAGES).
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
RUNTIME_REQUIREMENTS = [
    "torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0", "torchcodec==0.10.0", "torchao==0.16.0",
    "transformers>=4.51.0,<4.58.0", "diffusers>=0.37.0", "accelerate>=1.12.0", "safetensors==0.7.0",
    "einops>=0.8.1", "loguru>=0.7.3", "numba>=0.63.1", "vector-quantize-pytorch>=1.27.15", "scipy>=1.10.1",
    "soundfile>=0.13.1", "toml", "peft>=0.18.0", "pytorch-wavelets>=1.3.0", "pywavelets>=1.9.0", "diskcache",
    "xxhash", "pyyaml", "matplotlib>=3.7.5", "imageio-ffmpeg", "huggingface-hub", "hf_transfer",
    "mlflow==3.12.0", "sqlalchemy>=2.0,<2.1", "databricks-sdk>=0.102.0", "protobuf>=5.29.5,<6",
]
REGISTER_PIN_PACKAGES = [
    "torch", "torchvision", "torchaudio", "torchcodec", "torchao", "transformers", "diffusers", "accelerate",
    "safetensors", "einops", "loguru", "numba", "vector-quantize-pytorch", "scipy", "soundfile", "toml", "peft",
    "pytorch-wavelets", "PyWavelets", "diskcache", "xxhash", "PyYAML", "matplotlib", "imageio-ffmpeg",
    "huggingface-hub", "numpy", "pandas", "mlflow", "sqlalchemy", "tokenizers", "sentencepiece", "protobuf",
]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def prepare(work: str | Path = "/tmp/acestep") -> tuple[Path, Path]:
    """Clone ACE-Step at the pinned commit and download the checkpoints into `work`. Returns (repo, checkpoints).
    Use /tmp on serverless GPU: /local_disk0 is not writable there, and /tmp is the same large volume."""
    work = Path(work)
    repo, ckpt = work / "ACE-Step-1.5", work / "checkpoints"
    work.mkdir(parents=True, exist_ok=True)
    if not repo.is_dir():
        subprocess.run(["git", "clone", "-q", ACESTEP_REPO, str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "checkout", "-q", ACESTEP_SHA], check=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"   # hf_transfer is in RUNTIME_REQUIREMENTS
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    t0 = time.time()
    for hub_repo, sub in WEIGHTS:
        snapshot_download(hub_repo, local_dir=str(ckpt / sub) if sub else str(ckpt), max_workers=16)
    shutil.rmtree(ckpt / ".cache", ignore_errors=True)
    log(f"repo @ {ACESTEP_SHA[:7]}, checkpoints ready in {time.time() - t0:.0f}s")
    return repo, ckpt


def load_model(ckpt: str | Path, *, lm_model: str = LM_MODEL):
    """The serving PyFunc, loaded in-process (the same object the endpoint runs)."""
    if str(PYFUNC.parent) not in sys.path:
        sys.path.insert(0, str(PYFUNC.parent))
    import acestep_pyfunc

    class _Ctx:  # minimal stand-in for mlflow's PythonModelContext
        artifacts = {"checkpoints": str(ckpt)}
        model_config = {"dit_model": DIT_MODEL, "lm_model": lm_model, "lm_backend": "pt"}

    m = acestep_pyfunc.AceStepModel()
    t0 = time.time()
    m.load_context(_Ctx())
    log(f"load_context {time.time() - t0:.0f}s")
    return m


def run(model, action: str = "generate", params: dict[str, Any] | None = None, **audio_cols) -> dict:
    """One request through the PyFunc's predict: same columns as the endpoint (docs/SERVING-API.md).
    Returns the parsed result_json; raises RuntimeError with the endpoint's error text."""
    import pandas as pd
    row = {"action": action, "params_json": json.dumps(params or {}), **audio_cols}
    r = model.predict(None, pd.DataFrame([row]))
    if r["error"][0]:
        raise RuntimeError(r["error"][0])
    return json.loads(r["result_json"][0])
