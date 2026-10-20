#!/usr/bin/env bash
# Follow-up first pass (docs/followup/PREREG_AB.md) for ONE model on ONE GPU pod, separate
# from the main experiment: outputs go to followup_out/ and the HF prefix followup/ only;
# the main run's files are read, never written.
#
#   rp run <pod> --job followup --env SHUTDOWN=terminate --env MAX_HOURS=4 -- bash scripts/pod/run_followup.sh qwen3.6-27b
#
# Steps: preflight -> weights || EasySteer venv -> B generation (vLLM; throughput probe first)
# -> [B scoring + targeted judge on CPU] || [A ingest + cross-reading extraction on GPU]
# -> B analysis (discovery, then validation with the committed script) -> A analysis.
MODEL="${1:-qwen3.6-27b}"
MAX_HOURS="${MAX_HOURS:-4}"
export HF_LOG_PREFIX=followup/logs
# shellcheck source=scripts/pod/_lib.sh
source "$(dirname "$0")/_lib.sh"
pod_init followup "$MODEL"
OUT=followup_out
up() { upload "$1" --folder "$OUT/$2/$MODEL" --path-in-repo "followup/$2/$MODEL"; }

step "preflight"
doctor "$(need_gb)" --judge || finish 2

step "weights download (background) + EasySteer venv"
start_download
es_install
wait_download

step "B generation (720; throughput probe of 24 first)"
es_py -u scripts/followup/b_generate.py --model "$MODEL" --out "$OUT/B" --probe 24 > "$LOGDIR/b_generate.log" 2>&1 \
  || { tail -n 30 "$LOGDIR/b_generate.log"; finish 5; }
grep "\[B-gen\]" "$LOGDIR/b_generate.log"
up "B generations" B

step "B scoring + judge (CPU, background) || A cross-reading (GPU)"
$PY -u scripts/followup/b_score.py --model "$MODEL" --out "$OUT/B" > "$LOGDIR/b_score.log" 2>&1 &
BS=$!
$PY -u scripts/followup/a_crossread.py --model "$MODEL" --out "$OUT/A" > "$LOGDIR/a_crossread.log" 2>&1
A_RC=$?
grep "^\[A\]" "$LOGDIR/a_crossread.log" | tail -n 30
wait "$BS"; B_RC=$?
[ $B_RC = 0 ] || state "!! B scoring failed (see b_score.log)"
[ $A_RC = 0 ] || state "!! A cross-reading failed (see a_crossread.log)"

if [ $B_RC = 0 ]; then
  step "B analysis: discovery, then validation (committed script)"
  $PY scripts/followup/b_analyze.py --model "$MODEL" --out "$OUT/B" --split discovery | tee "$LOGDIR/b_analysis_discovery.txt"
  $PY scripts/followup/b_analyze.py --model "$MODEL" --out "$OUT/B" --split validation > "$LOGDIR/b_analysis_validation.txt" 2>&1
  up "B scores + analyses" B
fi
if [ $A_RC = 0 ]; then
  step "A analysis"
  $PY scripts/followup/a_analyze.py --model "$MODEL" --out "$OUT/A" | tee "$LOGDIR/a_analysis.txt"
  up "A cells + analysis" A
fi
{ [ $A_RC = 0 ] && [ $B_RC = 0 ]; } || finish 9
finish 0
