#!/usr/bin/env bash
# Q2 end to end for ONE model (after its Q1 run: needs outputs/<model>/analysis/vectors and
# betley_repro, on disk or pulled from $HF_DATASET_REPO with scripts/download_outputs.py).
#
#   rp run <pod> --job q2 -- bash scripts/pod/run_q2.sh qwen3.6-27b
#   SMOKE=1 bash scripts/pod/run_q2.sh qwen3.6-27b        # 4 prompts per benchmark, 2 strengths
#
# Steps: steering set (unit vectors + GGUF) -> EasySteer server -> sanity check ->
# calibration -> evaluation (client) -> stop server -> activation check (HF) ->
# School of Reward Hacks judges (API) -> behaviour report -> upload.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="${1:?usage: run_q2.sh <model-key>}"
set -a; [ -f .env ] && source .env; set +a
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
PY="${PY:-python}"
SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
LOGDIR="${LOGDIR:-/workspace/logs/$MODEL}"; mkdir -p "$LOGDIR"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s)-T0)/60 )) min] $*  $(date -u +%FT%TZ)"; }
upload() { [ -n "${HF_DATASET_REPO:-}" ] && [ -z "$SM" ] && $PY -u scripts/upload_hf.py --model "$MODEL" --message "$1" || true; }
finish() {
  bash scripts/pod/serve_easysteer.sh stop >/dev/null 2>&1
  step "done (exit $1)"
  if [ -n "${SHUTDOWN:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    [ "$SHUTDOWN" = terminate ] && runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
  fi
  exit "$1"
}

if [ ! -d "outputs/$MODEL/analysis/vectors" ] && [ -n "${HF_DATASET_REPO:-}" ]; then
  step "fetch Q1 outputs from $HF_DATASET_REPO"
  $PY -u scripts/download_outputs.py --model "$MODEL" --what analysis betley_repro || finish 2
fi
[ -f /workspace/.es_venv_path ] || bash scripts/pod/install_easysteer.sh > "$LOGDIR/install_es.log" 2>&1 || finish 3

step "steering set"
$PY -u scripts/10_build_steering_vectors.py --model "$MODEL" $SM || finish 4

step "EasySteer server"
bash scripts/pod/serve_easysteer.sh "$MODEL" || finish 5

step "sanity + calibration + evaluation"
$PY -u scripts/11_steer_generate.py --model "$MODEL" --phase all $SM 2>&1 | tee "$LOGDIR/steer_generate.log"
[ "${PIPESTATUS[0]}" = 0 ] || finish 6
bash scripts/pod/serve_easysteer.sh stop
upload "Q2 generations"

step "activation check (HF)"
$PY -u scripts/13_steer_activation_check.py --model "$MODEL" $SM > "$LOGDIR/activation_check.log" 2>&1 || \
  echo "!! activation check failed (see log)"

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  step "School of Reward Hacks judges"
  $PY -u scripts/12_steer_judge.py --model "$MODEL" $SM > "$LOGDIR/sorh_judge.log" 2>&1 || echo "!! judging failed (re-run 12_steer_judge.py)"
fi

step "behaviour report"
$PY -u scripts/14_behaviour_report.py --model "$MODEL" $SM || finish 7
upload "Q2 results"
finish 0
