#!/usr/bin/env bash
# Q1 end to end for ONE model on one pod (all GPUs used data-parallel). Idempotent and
# resumable: every step skips finished work, so re-running after a crash continues.
#
#   rp run <pod> --job q1 -- bash scripts/pod/run_q1.sh qwen3.6-27b
#   SMOKE_ONLY=1 bash scripts/pod/run_q1.sh qwen3.6-27b     # technical smoke test, then exit
#   STAGES="main" SKIP_SMOKE=1 bash scripts/pod/run_q1.sh qwen3.8-27b
#   SHUTDOWN=stop bash scripts/pod/run_q1.sh qwen3.6-27b    # runpodctl stop pod at the end
#
# Steps: weights download (background, hf_transfer) || EasySteer venv install -> smoke test
# (tiny grid, budget report) -> generation (vLLM, one worker per GPU) -> Betley
# reproduction + layer verification (HF) -> extraction (HF, one worker per GPU) || judge
# (API, background) -> analysis -> upload to $HF_DATASET_REPO after each step.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL="${1:?usage: run_q1.sh <model-key>}"
set -a; [ -f .env ] && source .env; set +a
export HF_HOME="${HF_HOME:-/workspace/hf}" HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAGES="${STAGES:-main paraphrase followup}"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l | tr -d ' ')}"
PY="${PY:-python}"
LOGDIR="${LOGDIR:-/workspace/logs/$MODEL}"; mkdir -p "$LOGDIR"
T0=$(date +%s)
step() { echo; echo "=== [$(( ($(date +%s)-T0)/60 )) min] $*  $(date -u +%FT%TZ)"; }
upload() { [ -n "${HF_DATASET_REPO:-}" ] && $PY -u scripts/upload_hf.py --model "$MODEL" --message "$1" || echo "(no HF_DATASET_REPO: skipping upload)"; }
finish() {
  step "done (exit $1)"
  if [ -n "${SHUTDOWN:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
    [ "$SHUTDOWN" = terminate ] && runpodctl remove pod "$RUNPOD_POD_ID" || runpodctl stop pod "$RUNPOD_POD_ID"
  fi
  exit "$1"
}

step "weights download (background) + EasySteer venv install, $NGPU GPU(s)"
$PY -u scripts/download_model.py --model "$MODEL" > "$LOGDIR/download.log" 2>&1 &
DL=$!
bash scripts/pod/install_easysteer.sh > "$LOGDIR/install_es.log" 2>&1 || { tail -30 "$LOGDIR/install_es.log"; finish 3; }
ES_PY="$(cat /workspace/.es_venv_path)/bin/python"
$PY -m pip install -q --no-deps "fla-core==0.5.2" "flash-linear-attention==0.5.2" einops 2>/dev/null \
  && echo "flash-linear-attention installed" || echo "(flash-linear-attention unavailable: HF uses the torch fallback)"
wait $DL || { echo "!! weight download failed"; tail -20 "$LOGDIR/download.log"; finish 4; }

if [ "${SKIP_SMOKE:-0}" != 1 ]; then
  step "smoke test (tiny grid, outputs_smoke/)"
  $ES_PY -u scripts/01_generate.py --model "$MODEL" --smoke --stages main paraphrase followup || finish 5
  $PY -u scripts/02_extract.py --model "$MODEL" --smoke --require-verification || finish 6
  $PY -u scripts/smoke_report.py --model "$MODEL" | tee "$LOGDIR/smoke_report.txt"
  [ -n "${OPENAI_API_KEY:-}" ] && $PY -u scripts/03_judge.py --model "$MODEL" --smoke
  $PY -u scripts/04_analyze.py --model "$MODEL" --smoke --reps pre resp --layers 36 --no-plots || finish 7
  [ "${SMOKE_ONLY:-0}" = 1 ] && finish 0
fi

step "generation: stages [$STAGES], $NGPU worker(s)"
pids=()
for ((i=0; i<NGPU; i++)); do
  CUDA_VISIBLE_DEVICES=$i $ES_PY -u scripts/01_generate.py --model "$MODEL" --stages $STAGES \
    --worker $i --num-workers "$NGPU" > "$LOGDIR/generate_$i.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { echo "!! a generation worker failed"; tail -n 20 "$LOGDIR"/generate_*.log; finish 8; }
upload "responses ($STAGES)"

if [ -n "${OPENAI_API_KEY:-}" ]; then
  step "judge (background, API)"
  $PY -u scripts/03_judge.py --model "$MODEL" > "$LOGDIR/judge.log" 2>&1 &
  JUDGE=$!
fi

step "Betley reproduction + layer/template verification"
VERIFY=""; [[ "$MODEL" == qwen3.6* ]] && VERIFY="--require-verification"
CUDA_VISIBLE_DEVICES=0 $PY -u scripts/02_extract.py --model "$MODEL" --betley-only $VERIFY || finish 9

step "activation extraction, $NGPU worker(s)"
pids=()
for ((i=0; i<NGPU; i++)); do
  CUDA_VISIBLE_DEVICES=$i $PY -u scripts/02_extract.py --model "$MODEL" --skip-betley \
    --worker $i --num-workers "$NGPU" > "$LOGDIR/extract_$i.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail = 0 ] || { echo "!! an extraction worker failed"; tail -n 20 "$LOGDIR"/extract_*.log; finish 10; }
[ -n "${JUDGE:-}" ] && { wait "$JUDGE" || echo "!! judge failed (see $LOGDIR/judge.log); re-run scripts/03_judge.py"; }
upload "activations + judge scores"

step "Q1 analysis"
$PY -u scripts/04_analyze.py --model "$MODEL" > "$LOGDIR/analysis.log" 2>&1 || { tail -30 "$LOGDIR/analysis.log"; finish 11; }
upload "Q1 analysis"
finish 0
