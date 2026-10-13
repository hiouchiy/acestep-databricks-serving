#!/bin/bash
# Install the ACE-Step 1.5 runtime on serverless GPU (environment version 4). torch 2.10 / cu128 replaces the
# preinstalled torch. Used by the notebooks and air_job/generate.yaml; the pins live in acestep_core.py
# (RUNTIME_REQUIREMENTS), and the serving registration pins the exact versions this install leaves.
#
#   bash scripts/install.sh                         # from the repo root (AIR job)
#   PYTHON=/path/to/python bash scripts/install.sh  # into a specific interpreter (the notebooks pass their own)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
INDEX=$("$PY" -c "import acestep_core; print(acestep_core.TORCH_INDEX)")
mapfile -t REQS < <("$PY" -c "import acestep_core; print('\n'.join(acestep_core.RUNTIME_REQUIREMENTS))")
"$PY" -m pip install -q --no-cache-dir --extra-index-url "$INDEX" "${REQS[@]}" 2>&1 | tail -30
"$PY" -m pip check || true
"$PY" -c "import torch; print('[install] torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# ffmpeg for mp3/opus output comes from imageio-ffmpeg (the PyFunc puts it on PATH).
