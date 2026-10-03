#!/usr/bin/env bash
# Upload the PyFunc + registration notebook to your workspace and run it as a one-off serverless job.
#   PROFILE=<cli-profile> UC_MODEL=<catalog>.<schema>.acestep_v15_xl_sft [COMPUTE=cpu|gpu] ./scripts/register.sh
# COMPUTE=cpu (default): download + bundle + register on CPU serverless (no smoke test; most reliable).
# COMPUTE=gpu: same on serverless GPU_1xA10, plus a real 30 s generation smoke test before registering.
set -euo pipefail
: "${PROFILE:?set PROFILE}"; : "${UC_MODEL:?set UC_MODEL=<catalog>.<schema>.<model>}"
COMPUTE="${COMPUTE:-cpu}"; LM_MODEL="${LM_MODEL:-acestep-5Hz-lm-1.7B}"
ME=$(databricks current-user me --profile "$PROFILE" -o json | python3 -c 'import sys,json;print(json.load(sys.stdin)["userName"])')
WS="/Workspace/Users/$ME/acestep_serving"
cd "$(dirname "$0")/.."
databricks workspace mkdirs "$WS" --profile "$PROFILE"
databricks workspace import "$WS/acestep_pyfunc.py" --file serving/acestep_pyfunc.py --format AUTO --overwrite --profile "$PROFILE"
databricks workspace import "$WS/register_acestep" --file serving/register_acestep.py --format SOURCE --language PYTHON --overwrite --profile "$PROFILE"
if [ "$COMPUTE" = gpu ]; then
  TASK_COMPUTE='"environment_key": "env", "compute": {"hardware_accelerator": "GPU_1xA10"}'
else
  TASK_COMPUTE='"environment_key": "env"'
fi
databricks jobs submit --no-wait --profile "$PROFILE" --json "{
  \"run_name\": \"acestep-register\", \"timeout_seconds\": 14400,
  \"tasks\": [{\"task_key\": \"register\",
    \"notebook_task\": {\"notebook_path\": \"$WS/register_acestep\",
      \"base_parameters\": {\"uc_model\": \"$UC_MODEL\", \"lm_model\": \"$LM_MODEL\", \"pyfunc_path\": \"$WS/acestep_pyfunc.py\"}},
    $TASK_COMPUTE}],
  \"environments\": [{\"environment_key\": \"env\", \"spec\": {\"client\": \"4\"}}]}"
echo "Track it with: databricks jobs get-run <run_id> --profile $PROFILE"
