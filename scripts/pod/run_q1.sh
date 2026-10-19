#!/usr/bin/env bash
# Q1 end to end for ONE model on one pod (all GPUs used data-parallel). Idempotent and
# resumable: every step skips finished work, so re-running after a crash continues.
#
# Launch (laptop, from the repo root; see README "Running on RunPod"):
#   rp up --name gs-q36 --gpu h200 --gpus 2 --volume none --volume-size 200 --disk 80
#   rp bootstrap gs-q36 --repo https://github.com/timf34/grader-shifts-persona-space --env .env --req requirements/pod.txt
#   rp run gs-q36 --job q1 --env SHUTDOWN=stop -- bash scripts/pod/run_q1.sh qwen3.6-27b
#   rp logs gs-q36 --job q1 -n 40          # or watch logs/q1/<model>/STATE.md on the HF dataset
#
# Knobs (env): SMOKE_ONLY=1 (smoke test, then exit) | SKIP_SMOKE=1 | DOCTOR_ONLY=1 |
#   RUN_ANALYSIS=1 (run 04_analyze.py on the GPU pod; default off: it is CPU-only) |
#   STAGES="main wording_validation followup" | GEN_TP=2 (tensor parallel per generation
#   worker, e.g. on 80 GB GPUs) | SHUTDOWN=stop|terminate | MAX_HOURS=48 | NGPU=<n>
#
# Steps: preflight -> weights download (background) || EasySteer venv -> smoke test (tiny
# grid, outputs_smoke/) -> generation (vLLM, one worker per GPU group) -> Betley
# reproduction + layer verification (HF) -> extraction (HF, one worker per GPU) || judge
# (API, background) -> analysis; upload to $HF_DATASET_REPO after each step.
MODEL="${1:?usage: run_q1.sh <model-key>}"
MAX_HOURS="${MAX_HOURS:-48}"
# shellcheck source=scripts/pod/_lib.sh
source "$(dirname "$0")/_lib.sh"
pod_init q1 "$MODEL"
STAGES="${STAGES:-main wording_validation followup}"
JUDGE_FLAG=""; judge_ready && JUDGE_FLAG="--judge"
TP_FLAG=""; [ "$GEN_TP" -gt 1 ] && TP_FLAG="--tensor-parallel-size $GEN_TP"
# Betley et al. published their vector for Qwen3.6-27B only: a hard verification there; for
# other models the comparison is a cross-model reference, never a pass/fail gate
VERIFY=""; [[ "$MODEL" == qwen3.6* ]] && VERIFY="--require-verification"

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
  es_py -u scripts/01_generate.py --model "$MODEL" --smoke --stages main wording_validation followup $TP_FLAG \
    > "$LOGDIR/smoke_generate.log" 2>&1 || { tail -n 30 "$LOGDIR/smoke_generate.log"; finish 5; }
  # shellcheck disable=SC2086
  $PY -u scripts/02_extract.py --model "$MODEL" --smoke $VERIFY > "$LOGDIR/smoke_extract.log" 2>&1 \
    || { tail -n 30 "$LOGDIR/smoke_extract.log"; finish 6; }
  $PY -u scripts/smoke_report.py --model "$MODEL" | tee "$LOGDIR/smoke_report.txt"
  $PY -u scripts/gpu_checks.py spans --model "$MODEL" --smoke | tee "$LOGDIR/smoke_spans.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || finish 6
  CUDA_VISIBLE_DEVICES=0 $PY -u scripts/gpu_checks.py replay --model "$MODEL" --smoke | tee "$LOGDIR/smoke_replay.txt"
  [ "${PIPESTATUS[0]}" = 0 ] || finish 6
  [ -n "$JUDGE_FLAG" ] && $PY -u scripts/03_judge.py --model "$MODEL" --smoke > "$LOGDIR/smoke_judge.log" 2>&1
  $PY -u scripts/04_analyze.py --model "$MODEL" --smoke --reps pre resp --layers 36 --no-plots \
    > "$LOGDIR/smoke_analysis.log" 2>&1 || { tail -n 30 "$LOGDIR/smoke_analysis.log"; finish 7; }
  state "smoke test passed"
  upload_logs || true
  [ "${SMOKE_ONLY:-0}" = 1 ] && finish 0
fi

step "generation: stages [$STAGES], $NGPU GPU(s), TP $GEN_TP"
# shellcheck disable=SC2086
fanout generate "$GEN_TP" es_py scripts/01_generate.py --model "$MODEL" --stages $STAGES $TP_FLAG || finish 8
upload "responses ($STAGES)"

if [ -n "$JUDGE_FLAG" ]; then
  step "judge (background, API)"
  $PY -u scripts/03_judge.py --model "$MODEL" > "$LOGDIR/judge.log" 2>&1 &
  JUDGE=$!
fi

step "Betley reproduction + layer/template verification"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/02_extract.py --model "$MODEL" --betley-only $VERIFY \
  > "$LOGDIR/betley.log" 2>&1 || { tail -n 30 "$LOGDIR/betley.log"; finish 9; }
grep "^\[betley\]" "$LOGDIR/betley.log" | tail -n 1

step "activation extraction, $NGPU worker(s)"
fanout extract 1 "$PY" scripts/02_extract.py --model "$MODEL" --skip-betley || finish 10
if [ -n "${JUDGE:-}" ]; then
  wait "$JUDGE" || state "!! judge failed (see judge.log); re-run scripts/03_judge.py, then the analysis"
fi
upload "activations + judge scores"

# The CPU-only analysis (100 roles x 7 layers x 3 representations) would leave every GPU
# idle for hours: by default the GPU pod stops here and the analysis runs from the HF
# dataset elsewhere (scripts/download_outputs.py --what all, then scripts/04_analyze.py).
if [ "${RUN_ANALYSIS:-0}" = 1 ]; then
  step "Q1 analysis"
  $PY -u scripts/04_analyze.py --model "$MODEL" > "$LOGDIR/analysis.log" 2>&1 || { tail -n 30 "$LOGDIR/analysis.log"; finish 11; }
  grep "^\[analysis\]" "$LOGDIR/analysis.log" | grep " L36:" | head -n 6
  upload "Q1 analysis"
else
  state "generation + extraction + judge complete; analysis skipped on the GPU pod (RUN_ANALYSIS=1 to run it here)"
fi
finish 0
