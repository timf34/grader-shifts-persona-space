#!/usr/bin/env bash
# PSM decisions experiment, one model, one stage (pilot | confirm). After prep_psm.sh.
#
#   rp run <pod> --job psmpilot -- bash scripts/pod/run_psm.sh qwen3.6-27b pilot
#   rp run <pod> --job psmconfirm --env SHUTDOWN=stop -- bash scripts/pod/run_psm.sh qwen3.6-27b confirm
#
# Steps: git pull (frozen settings arrive by commit) -> preflight -> EasySteer venv (idempotent)
# -> HF phase (trait directions, prompt activations) -> vectors/GGUF -> server -> generation
# (checkpoint uploads) -> [pilot: pre-registered steering choice] -> upload. OLMo confirm:
# the steering choice comes from that model's own pilot sweep (select step), not from Qwen.
# Knobs: TRANSFER_FROM=<model key> (add that model's trait direction as traitdpo, OLMo final),
#        LIMIT=n (smoke), CONCURRENCY, PARALLEL, MAX_NUM_SEQS.
MODEL="${1:?usage: run_psm.sh <model-key> <pilot|confirm>}"
STAGE="${2:?stage}"
MAX_HOURS="${MAX_HOURS:-10}"
# shellcheck source=scripts/pod/_lib.sh
source "$(dirname "$0")/_lib.sh"
pod_init "psm$STAGE" "$MODEL"
git pull -q --ff-only 2>&1 | tail -n 2; state "repo at $(git rev-parse --short HEAD)"
FAM=$($PY - "$MODEL" <<'PY'
import sys; print("olmo" if sys.argv[1].startswith("olmo") else "qwen")
PY
)
export MAX_MODEL_LEN=$($PY - "$FAM" <<'PY'
import sys, yaml; print(yaml.safe_load(open("configs/psm.yaml"))["generation"][sys.argv[1]]["max_model_len"])
PY
)
LIM=""; [ -n "${LIMIT:-}" ] && LIM="--limit $LIMIT"
TF=""; [ -n "${TRANSFER_FROM:-}" ] && TF="--transfer-from $TRANSFER_FROM"
psm_upload() {
  upload "$1" --folder "outputs/$MODEL/psm" --path-in-repo "psm/$MODEL"
}

step "preflight"
doctor 40 || finish 2
step "EasySteer venv (idempotent) + weights"
start_download
es_install
wait_download
[ -f "outputs/$MODEL/betley_repro/report.json" ] || $PY -u scripts/download_outputs.py --model "$MODEL" --what betley_repro || finish 2
if [ -n "${TRANSFER_FROM:-}" ]; then
  $PY - "$TRANSFER_FROM" <<'PY' || state "!! transfer direction not available yet"
import os, sys
from huggingface_hub import hf_hub_download
m = sys.argv[1]
p = hf_hub_download(os.environ["HF_DATASET_REPO"], f"psm/{m}/hf/trait_dirs_ending.npz", repo_type="dataset",
                    local_dir="outputs_tmp_transfer")
os.makedirs(f"outputs/{m}/psm/hf", exist_ok=True)
os.replace(p, f"outputs/{m}/psm/hf/trait_dirs_ending.npz")
print("transfer direction from", m)
PY
fi
if [ "$STAGE" = confirm ] && [ ! -f "outputs/$MODEL/psm/steer_choice.json" ]; then
  $PY - "$MODEL" <<'PY' || finish 2
import os, sys
from huggingface_hub import hf_hub_download
m = sys.argv[1]
p = hf_hub_download(os.environ["HF_DATASET_REPO"], f"psm/{m}/steer_choice.json", repo_type="dataset", local_dir="outputs_tmp_choice")
os.makedirs(f"outputs/{m}/psm", exist_ok=True); os.replace(p, f"outputs/{m}/psm/steer_choice.json"); print("steer choice from HF")
PY
fi

step "HF phase: trait directions + prompt activations ($STAGE)"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/psm/pod.py hf --model "$MODEL" --stage "$STAGE" 2>&1 | tee "$LOGDIR/hf.log"
[ "${PIPESTATUS[0]}" = 0 ] || finish 4
$PY -u scripts/psm/pod.py vectors --model "$MODEL" $TF 2>&1 | tee "$LOGDIR/vectors.log"
[ "${PIPESTATUS[0]}" = 0 ] || finish 4
psm_upload "psm $MODEL $STAGE: HF phase"

step "EasySteer server (max_model_len $MAX_MODEL_LEN)"
ES_SERVER_STARTED=1
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}" bash scripts/pod/serve_easysteer.sh "$MODEL" || finish 5

step "generation ($STAGE)"
( while sleep 900; do psm_upload "psm $MODEL $STAGE: generations (checkpoint)"; done ) &
CKPT_PID=$!
$PY -u scripts/psm/pod.py gen --model "$MODEL" --stage "$STAGE" --concurrency "${CONCURRENCY:-224}" \
  --parallel "${PARALLEL:-4}" $LIM $TF 2>&1 | tee "$LOGDIR/gen.log"
GEN_RC=${PIPESTATUS[0]}
kill "$CKPT_PID" 2>/dev/null; wait "$CKPT_PID" 2>/dev/null
bash scripts/pod/serve_easysteer.sh stop; ES_SERVER_STARTED=0
[ "$GEN_RC" = 0 ] || { psm_upload "psm $MODEL $STAGE: partial generations"; finish 6; }

if [ "$STAGE" = pilot ]; then
  step "pre-registered steering choice"
  $PY -u scripts/psm/pod.py select --model "$MODEL" 2>&1 | tee "$LOGDIR/select.log"
fi
psm_upload "psm $MODEL $STAGE: done"
finish 0
