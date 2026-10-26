#!/usr/bin/env bash
# PSM: steering-sweep extension (pilot stage, steer_base only, then the pre-registered choice) followed
# by confirmation, in one job. SHUTDOWN (if set) applies only after confirmation.
#   rp run <pod> --job psmchain --env SHUTDOWN=stop [--env TRANSFER_FROM=olmo3-32b-dpo] -- bash scripts/pod/run_psm_chain.sh <model>
MODEL="${1:?model}"
cd "$(dirname "$0")/../.." || exit 1
FINAL_SHUTDOWN="${SHUTDOWN:-}"
SHUTDOWN="" ONLY=steer_base TRANSFER_FROM="" LOGDIR="/workspace/logs/psmext-$MODEL" bash scripts/pod/run_psm.sh "$MODEL" pilot \
  || echo "!! sweep extension failed; confirmation uses the existing steer_choice.json"
SHUTDOWN="$FINAL_SHUTDOWN" LOGDIR="/workspace/logs/psmconfirm-$MODEL" bash scripts/pod/run_psm.sh "$MODEL" confirm
