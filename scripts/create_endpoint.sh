#!/usr/bin/env bash
# Create (or re-point) the serving endpoint.
#   PROFILE=<cli-profile> UC_MODEL=<catalog>.<schema>.acestep_v15_xl_sft VERSION=<n> ./scripts/create_endpoint.sh
# Optional: ENDPOINT (default acestep-v15-xl-sft), WORKLOAD (default GPU_MEDIUM = 1x A10 24 GB),
#           SCALE_TO_ZERO (default true), LM_MODEL (sets env ACESTEP_LM_MODEL; 4B needs a >24 GB GPU).
set -euo pipefail
: "${PROFILE:?}"; : "${UC_MODEL:?}"; : "${VERSION:?}"
ENDPOINT="${ENDPOINT:-acestep-v15-xl-sft}"; WORKLOAD="${WORKLOAD:-GPU_MEDIUM}"; S2Z="${SCALE_TO_ZERO:-true}"
NAME="$(basename "${UC_MODEL//./\/}")-$VERSION"
ENV_VARS=""
[ -n "${LM_MODEL:-}" ] && ENV_VARS=", \"environment_vars\": {\"ACESTEP_LM_MODEL\": \"$LM_MODEL\"}"
CFG="{\"served_entities\": [{\"name\": \"$NAME\", \"entity_name\": \"$UC_MODEL\", \"entity_version\": \"$VERSION\",
  \"workload_type\": \"$WORKLOAD\", \"workload_size\": \"Small\", \"scale_to_zero_enabled\": $S2Z $ENV_VARS}],
  \"traffic_config\": {\"routes\": [{\"served_model_name\": \"$NAME\", \"traffic_percentage\": 100}]}}"
if databricks serving-endpoints get "$ENDPOINT" --profile "$PROFILE" >/dev/null 2>&1; then
  databricks serving-endpoints update-config "$ENDPOINT" --no-wait --profile "$PROFILE" --json "$CFG"
else
  databricks serving-endpoints create --no-wait --profile "$PROFILE" --json "{\"name\": \"$ENDPOINT\", \"config\": $CFG}"
fi
echo "Poll: databricks serving-endpoints get $ENDPOINT --profile $PROFILE | jq '.state'   (first build ~1 h: ~36 GB image)"
