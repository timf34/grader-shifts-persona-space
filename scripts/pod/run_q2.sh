#!/usr/bin/env bash
# Q2 end to end for ONE model (after its Q1 run: needs outputs/<model>/analysis/vectors and
# betley_repro, on disk or pulled from $HF_DATASET_REPO with scripts/download_outputs.py).
#
#   rp run <pod> --job q2 --env SHUTDOWN=stop -- bash scripts/pod/run_q2.sh qwen3.6-27b
#   SMOKE=1 bash scripts/pod/run_q2.sh qwen3.6-27b        # 4 prompts per benchmark, 2 strengths
#
# Steps: preflight -> weights || EasySteer venv -> steering set (unit vectors + GGUF) ->
# steering-path validation BEFORE any evaluation (server: zero strength == unsteered;
# HF: the server's steered log-probs match strength x A_L x unit(v) at OUR layer L, not
# L+-1 / x0.5 / x2 / the opposite sign; GGUF round trip) -> server -> calibration +
# evaluation (client) -> stop server -> activation check (HF) -> School of Reward Hacks
# judges (API) -> behaviour report; upload after each step.
MODEL="${1:?usage: run_q2.sh <model-key>}"
MAX_HOURS="${MAX_HOURS:-24}"
# shellcheck source=scripts/pod/_lib.sh
source "$(dirname "$0")/_lib.sh"
pod_init q2 "$MODEL"
SM=""; [ "${SMOKE:-0}" = 1 ] && SM="--smoke"
q2_upload() { [ -z "$SM" ] && upload "$@"; return 0; }

step "preflight"
doctor "$(need_gb)" || finish 2
[ "${DOCTOR_ONLY:-0}" = 1 ] && finish 0

if [ ! -d "outputs/$MODEL/analysis/vectors" ] && [ -n "${HF_DATASET_REPO:-}" ]; then
  step "fetch Q1 outputs from $HF_DATASET_REPO"
  $PY -u scripts/download_outputs.py --model "$MODEL" --what analysis betley_repro || finish 2
fi

step "weights download (background) + EasySteer venv"
start_download
es_install
wait_download

step "steering set"
$PY -u scripts/10_build_steering_vectors.py --model "$MODEL" $SM || finish 4

step "steering-path validation (server vs HF)"
ES_SERVER_STARTED=1
bash scripts/pod/serve_easysteer.sh "$MODEL" || finish 5
$PY -u scripts/gpu_checks.py steer-server --model "$MODEL" | tee "$LOGDIR/steer_server_check.txt"
[ "${PIPESTATUS[0]}" = 0 ] || finish 5
bash scripts/pod/serve_easysteer.sh stop
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/gpu_checks.py steer-hf --model "$MODEL" | tee "$LOGDIR/steer_hf_check.txt"
[ "${PIPESTATUS[0]}" = 0 ] || finish 5
state "steering path verified (layer, positions, sign, scale, GGUF)"

step "EasySteer server"
bash scripts/pod/serve_easysteer.sh "$MODEL" || finish 5

step "sanity + calibration + evaluation"
$PY -u scripts/11_steer_generate.py --model "$MODEL" --phase all $SM 2>&1 | tee "$LOGDIR/steer_generate.log"
[ "${PIPESTATUS[0]}" = 0 ] || finish 6
bash scripts/pod/serve_easysteer.sh stop
ES_SERVER_STARTED=0
q2_upload "Q2 generations"

step "activation check (HF)"
$PY -u scripts/13_steer_activation_check.py --model "$MODEL" $SM > "$LOGDIR/activation_check.log" 2>&1 || \
  state "!! activation check failed (see activation_check.log)"

if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
  step "judges (School of Reward Hacks, trait rubric)"
  $PY -u scripts/12_steer_judge.py --model "$MODEL" $SM > "$LOGDIR/steer_judge.log" 2>&1 || \
    state "!! judging failed (re-run scripts/12_steer_judge.py)"
fi

step "behaviour report"
$PY -u scripts/14_behaviour_report.py --model "$MODEL" $SM > "$LOGDIR/behaviour_report.log" 2>&1 \
  || { tail -n 30 "$LOGDIR/behaviour_report.log"; finish 7; }
q2_upload "Q2 results"
finish 0
