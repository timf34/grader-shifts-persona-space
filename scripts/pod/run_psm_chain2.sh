#!/usr/bin/env bash
# As run_psm_chain.sh, but EXT_ONLY selects the extension stage's benchmarks:
# EXT_ONLY=all completes every pilot job (e.g. after a crashed pilot) before the sweep extension.
#   rp run <pod> --job psmchain --env SHUTDOWN=stop --env EXT_ONLY=all -- bash scripts/pod/run_psm_chain2.sh <model>
MODEL="${1:?model}"
cd "$(dirname "$0")/../.." || exit 1
FINAL_SHUTDOWN="${SHUTDOWN:-}"
ONLY_VAL="${EXT_ONLY:-steer_base}"; [ "$ONLY_VAL" = all ] && ONLY_VAL=""
SHUTDOWN="" ONLY="$ONLY_VAL" TRANSFER_FROM="" LOGDIR="/workspace/logs/psmext-$MODEL" bash scripts/pod/run_psm.sh "$MODEL" pilot \
  || echo "!! pilot completion / sweep extension failed; confirmation uses the existing steer_choice.json"
SHUTDOWN="$FINAL_SHUTDOWN" LOGDIR="/workspace/logs/psmconfirm-$MODEL" bash scripts/pod/run_psm.sh "$MODEL" confirm
