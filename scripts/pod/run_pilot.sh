#!/usr/bin/env bash
# Discovery-only thinking ON / OFF pilot for ONE model (configs/experiment.yaml `pilot:`):
# 10 discovery roles x 3 prompt variants x 20 discovery questions + default control =
# 2,480 generations per arm. Each arm has its own output root
# (outputs_pilot/thinking_<on|off>/) and config id; nothing here feeds the confirmatory
# analysis. Idempotent and resumable like run_q1.sh.
#
# Launch (laptop, from the repo root; one pod per model, or run the two models in turn):
#   rp up --name gs-pilot36 --gpu h200 --gpus 1 --volume none --volume-size 200 --disk 80
#   rp bootstrap gs-pilot36 --repo https://github.com/timf34/grader-shifts-persona-space --env .env --req requirements/pod.txt
#   rp run gs-pilot36 --job pilot --env SHUTDOWN=stop -- bash scripts/pod/run_pilot.sh qwen3.6-27b
#   rp logs gs-pilot36 --job pilot -n 40
#
# Knobs (env): SKIP_SMOKE=1 | DOCTOR_ONLY=1 | ARMS="on off" | GEN_TP=2 | SHUTDOWN=stop|terminate |
#   MAX_HOURS=10 | NGPU=<n>
#
# Steps: preflight -> weights || EasySteer venv -> smoke test -> per arm: generation ->
# extraction (Betley verification first, thinking-ON arm) || judge -> pilot report
# (truncation, token budgets, timings + GPU-hour estimate, adherence, displacement
# reliability, thinking sensitivity); each arm uploaded to pilot/thinking_<arm>/<model>/.
MODEL="${1:?usage: run_pilot.sh <model-key>}"
MAX_HOURS="${MAX_HOURS:-10}"
# shellcheck source=scripts/pod/_lib.sh
source "$(dirname "$0")/_lib.sh"
pod_init "${RP_JOB:-pilot}" "$MODEL"
ARMS="${ARMS:-on off}"
JUDGE_FLAG=""; [ -n "${OPENAI_API_KEY:-}" ] && JUDGE_FLAG="--judge"
TP_FLAG=""; [ "$GEN_TP" -gt 1 ] && TP_FLAG="--tensor-parallel-size $GEN_TP"

step "preflight"
doctor "$(need_gb)" $JUDGE_FLAG || finish 2
[ "${DOCTOR_ONLY:-0}" = 1 ] && finish 0

step "weights download (background) + EasySteer venv"
start_download
es_install
wait_download

if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  step "smoke test (tiny grid, outputs_smoke/)"
  # shellcheck disable=SC2086
  es_py -u scripts/01_generate.py --model "$MODEL" --smoke --stages main $TP_FLAG > "$LOGDIR/smoke_generate.log" 2>&1 \
    || { tail -n 30 "$LOGDIR/smoke_generate.log"; finish 5; }
  $PY -u scripts/02_extract.py --model "$MODEL" --smoke --require-verification > "$LOGDIR/smoke_extract.log" 2>&1 \
    || { tail -n 30 "$LOGDIR/smoke_extract.log"; finish 6; }
  $PY -u scripts/smoke_report.py --model "$MODEL" | tee "$LOGDIR/smoke_report.txt"
  $PY -u scripts/gpu_checks.py spans --model "$MODEL" --smoke | tee "$LOGDIR/smoke_spans.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || finish 6
  CUDA_VISIBLE_DEVICES=0 $PY -u scripts/gpu_checks.py replay --model "$MODEL" --smoke | tee "$LOGDIR/smoke_replay.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || finish 6
  state "smoke test passed"
  upload_logs || true
fi

for arm in $ARMS; do
  step "pilot thinking $arm: generation"
  # shellcheck disable=SC2086
  fanout "generate_$arm" "$GEN_TP" es_py scripts/01_generate.py --model "$MODEL" --stages pilot --thinking "$arm" \
    $TP_FLAG || finish 8
  upload "pilot thinking_$arm responses" --folder "outputs_pilot/thinking_$arm/$MODEL" \
    --path-in-repo "pilot/thinking_$arm/$MODEL"
  JUDGE=""
  if [ -n "$JUDGE_FLAG" ]; then
    $PY -u scripts/03_judge.py --model "$MODEL" --thinking "$arm" > "$LOGDIR/judge_$arm.log" 2>&1 &
    JUDGE=$!
  fi
  if [ "$arm" = on ]; then
    step "Betley reproduction + layer/template verification (thinking on)"
    VERIFY=""; [[ "$MODEL" == qwen3.6* ]] && VERIFY="--require-verification"
    CUDA_VISIBLE_DEVICES=0 $PY -u scripts/02_extract.py --model "$MODEL" --thinking on --betley-only $VERIFY \
      > "$LOGDIR/betley.log" 2>&1 || { tail -n 30 "$LOGDIR/betley.log"; finish 9; }
    grep "^\[betley\]" "$LOGDIR/betley.log" | tail -n 1
  fi
  step "pilot thinking $arm: extraction"
  fanout "extract_$arm" 1 "$PY" scripts/02_extract.py --model "$MODEL" --thinking "$arm" --skip-betley || finish 10
  if [ -n "$JUDGE" ]; then
    wait "$JUDGE" || state "!! judge ($arm) failed (see judge_$arm.log)"
  fi
  upload "pilot thinking_$arm activations + scores" --folder "outputs_pilot/thinking_$arm/$MODEL" \
    --path-in-repo "pilot/thinking_$arm/$MODEL"
done

step "pilot report"
$PY -u scripts/pilot_report.py --model "$MODEL" --json | tee "$LOGDIR/pilot_report.txt"
[ "${PIPESTATUS[0]}" = 0 ] || finish 11
cp "outputs_pilot/${MODEL}_pilot_report.json" "$LOGDIR/" 2>/dev/null
finish 0
