#!/usr/bin/env bash
# Q2 end to end for ONE model (after its Q1 run: needs outputs/<model>/analysis/vectors and
# betley_repro, on disk or pulled from $HF_DATASET_REPO with scripts/download_outputs.py).
#
#   rp run <pod> --job q2 --env SHUTDOWN=stop -- bash scripts/pod/run_q2.sh qwen3.6-27b
#   SMOKE=1 bash scripts/pod/run_q2.sh qwen3.6-27b        # 4 prompts per benchmark, 2 strengths
#   STEERING_CONFIG=configs/steering_stage_a.yaml SHARD=0 SHARDS=3 SKIP_JUDGE=1 PROBES=1 \
#     bash scripts/pod/run_q2.sh qwen3.6-27b              # stage A, one of three pods
#
# Knobs (env): STEERING_CONFIG (default configs/steering.yaml) | SHARD=k SHARDS=m (m pods split the
#   steering conditions round-robin; same model) | SKIP_JUDGE=1 (judge later from a laptop) |
#   SKIP_ACTCHECK=1 | SKIP_VALIDATION=1 (skip the server-vs-HF steering-path check) | PROBES=1
#   (context-probe forward passes, scripts/15_context_probes.py, before the server starts) |
#   PARALLEL_CONDITIONS (default 3) | CHECKPOINT_MIN (default 15: upload generations while running).
#
# Steps: preflight -> weights || EasySteer venv -> steering set (unit vectors + GGUF) ->
# [context probes, HF] -> steering-path validation BEFORE any evaluation (server: zero strength ==
# unsteered; HF: the server's steered log-probs match strength x A_L x unit(v) at OUR layer L,
# not L+-1 / x0.5 / x2 / the opposite sign; GGUF round trip) -> server -> [calibration] +
# evaluation (client) -> stop server -> [activation check (HF)] -> [judges (API)] ->
# [behaviour report]; upload after each step and every CHECKPOINT_MIN minutes while generating.
MODEL="${1:?usage: run_q2.sh <model-key>}"
MAX_HOURS="${MAX_HOURS:-24}"
# shellcheck source=scripts/pod/_lib.sh
source "$(dirname "$0")/_lib.sh"
pod_init q2 "$MODEL"
SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
export STEERING_CONFIG="${STEERING_CONFIG:-configs/steering.yaml}"
SC=(--steering-config "$STEERING_CONFIG")
q2_upload() {  # steering outputs (+ probes) only; the Q1 inputs on this pod are already on HF
  [ -n "$SM" ] && return 0
  upload "$1" --folder "outputs/$MODEL/steering" --path-in-repo "$MODEL/steering"
  [ -d "outputs/$MODEL/probes" ] && upload "$1 (probes)" --folder "outputs/$MODEL/probes" --path-in-repo "$MODEL/probes"
  return 0
}

step "preflight"
doctor "$(need_gb)" || finish 2
[ "${DOCTOR_ONLY:-0}" = 1 ] && finish 0
state "steering config $STEERING_CONFIG, shard $SHARD/$SHARDS"

if [ ! -d "outputs/$MODEL/analysis/vectors" ] && [ -n "${HF_DATASET_REPO:-}" ]; then
  step "fetch Q1 outputs from $HF_DATASET_REPO"
  $PY -u scripts/download_outputs.py --model "$MODEL" --what analysis betley_repro || finish 2
fi
# STEER_SOURCE_MODELS: models whose Q1 directions the steering config borrows (source_model)
for SRC in ${STEER_SOURCE_MODELS:-}; do
  [ "$SRC" = "$MODEL" ] && continue
  [ -d "outputs/$SRC/analysis/vectors" ] || $PY -u scripts/download_outputs.py --model "$SRC" --what analysis || finish 2
done

step "weights download (background) + EasySteer venv"
start_download
es_install
wait_download

step "steering set"
$PY -u scripts/10_build_steering_vectors.py --model "$MODEL" "${SC[@]}" $SM || finish 4

if [ "${PROBES:-0}" = 1 ]; then
  step "context probes (HF forward passes)"
  CUDA_VISIBLE_DEVICES=0 $PY -u scripts/15_context_probes.py --model "$MODEL" $SM > "$LOGDIR/probes.log" 2>&1 \
    && q2_upload "Q2 context probes" || state "!! context probes failed (see probes.log); continuing"
fi

if [ "${SKIP_VALIDATION:-0}" != 1 ]; then
  step "steering-path validation (server vs HF)"
  ES_SERVER_STARTED=1
  bash scripts/pod/serve_easysteer.sh "$MODEL" || finish 5
  $PY -u scripts/gpu_checks.py steer-server --model "$MODEL" | tee "$LOGDIR/steer_server_check.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || finish 5
  bash scripts/pod/serve_easysteer.sh stop
  CUDA_VISIBLE_DEVICES=0 $PY -u scripts/gpu_checks.py steer-hf --model "$MODEL" | tee "$LOGDIR/steer_hf_check.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || finish 5
  state "steering path verified (layer, positions, sign, scale, GGUF)"
fi

step "EasySteer server"
ES_SERVER_STARTED=1
bash scripts/pod/serve_easysteer.sh "$MODEL" || finish 5

step "sanity + calibration + evaluation (shard $SHARD/$SHARDS)"
( while sleep "$(( ${CHECKPOINT_MIN:-15} * 60 ))"; do q2_upload "Q2 generations (checkpoint)"; done ) &
CKPT_PID=$!
$PY -u scripts/11_steer_generate.py --model "$MODEL" --phase all "${SC[@]}" --shard "$SHARD" --shards "$SHARDS" \
  --parallel-conditions "${PARALLEL_CONDITIONS:-3}" $SM 2>&1 | tee "$LOGDIR/steer_generate.log"
GEN_RC=${PIPESTATUS[0]}
kill "$CKPT_PID" 2>/dev/null; wait "$CKPT_PID" 2>/dev/null
bash scripts/pod/serve_easysteer.sh stop
ES_SERVER_STARTED=0
q2_upload "Q2 generations"
[ "$GEN_RC" = 0 ] || finish 6

if [ "${SKIP_ACTCHECK:-0}" != 1 ]; then
  step "activation check (HF)"
  $PY -u scripts/13_steer_activation_check.py --model "$MODEL" "${SC[@]}" $SM > "$LOGDIR/activation_check.log" 2>&1 || \
    state "!! activation check failed (see activation_check.log)"
fi

if [ "${SKIP_JUDGE:-0}" != 1 ] && { [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${OPENROUTER_API_KEY:-}" ]; }; then
  step "judges"
  $PY -u scripts/12_steer_judge.py --model "$MODEL" "${SC[@]}" $SM > "$LOGDIR/steer_judge.log" 2>&1 || \
    state "!! judging failed (re-run scripts/12_steer_judge.py)"
  step "behaviour report"
  $PY -u scripts/14_behaviour_report.py --model "$MODEL" $SM > "$LOGDIR/behaviour_report.log" 2>&1 \
    || { tail -n 30 "$LOGDIR/behaviour_report.log"; finish 7; }
fi
q2_upload "Q2 results"
finish 0
