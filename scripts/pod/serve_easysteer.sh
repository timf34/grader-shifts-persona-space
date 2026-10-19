#!/usr/bin/env bash
# Start an EasySteer vLLM server for one model (per-request `direct` steering specs, no
# engine default), pinned revision and chat template from configs/models.yaml. Blocks until
# /v1/models answers. Adapted from timf34/AttractorStatePrefillAttack capped/easysteer/serve.sh.
#
#   bash scripts/pod/serve_easysteer.sh qwen3.6-27b      # -> http://localhost:8000/v1
#   bash scripts/pod/serve_easysteer.sh stop
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
# venv + compat libraries chosen for THIS host by install_easysteer.sh (re-run on every pod start)
[ -f /workspace/.es_env ] && . /workspace/.es_env
ES_LD_LIBRARY_PATH="${ES_LD_LIBRARY_PATH:-}"
PORT="${PORT:-8000}"
LOG="${LOG:-/workspace/es_server.log}"
export HF_HOME="${HF_HOME:-/workspace/hf}"

stop_server() {
  # only the server and its engine-core children (never e.g. a running `pip install vllm`)
  pkill -f "[v]llm serve" 2>/dev/null || true
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1 | tr -d ' ')
    { [ -z "$used" ] || [ "$used" -lt 2000 ]; } && return 0
    pkill -9 -f "[v]llm serve|[V]LLM::" 2>/dev/null || true; sleep 4
  done
}
[ "${1:-}" = "stop" ] && { stop_server; echo "server stopped"; exit 0; }
ES_VENV="${ES_VENV:?run scripts/pod/install_easysteer.sh first}"
MODEL_KEY="${1:?model key}"
read -r HF REV MAXLEN <<<"$("$ES_VENV/bin/python" - "$MODEL_KEY" "$REPO" <<'PY'
import sys, yaml
key, repo = sys.argv[1], sys.argv[2]
m = yaml.safe_load(open(f"{repo}/configs/models.yaml"))["models"][key]
s = yaml.safe_load(open(f"{repo}/configs/steering.yaml"))["generation"]
print(m["hf_id"], m["revision"], s["max_model_len"])
PY
)"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
TP="${TP:-1}"
GRAPH_MODE="${STEER_GRAPH_MODE:-split}"     # verified mode in the AttractorStatePrefillAttack runs
stop_server
PATH="$ES_VENV/bin:$PATH" LD_LIBRARY_PATH="${ES_LD_LIBRARY_PATH}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  nohup "$ES_VENV/bin/vllm" serve "$HF" --revision "$REV" --served-model-name "$MODEL_KEY" \
  --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --max-num-seqs "$MAX_NUM_SEQS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" --port "$PORT" --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enable-steer-vector --steer-algorithms direct --steer-graph-mode "$GRAPH_MODE" \
  --disable-custom-all-reduce > "$LOG" 2>&1 &
echo "vllm serve pid $! (log $LOG)"
SERVE_WAIT_MIN="${SERVE_WAIT_MIN:-60}"   # covers a cold 55 GB download + load
for i in $(seq 1 $((SERVE_WAIT_MIN*12))); do
  curl -sf "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$MODEL_KEY" && { echo "server up after ~$((i*5))s: http://localhost:$PORT/v1"; exit 0; }
  pgrep -f "[v]llm serve" >/dev/null || { echo "!! server died"; tail -n 40 "$LOG"; exit 1; }
  sleep 5
done
echo "!! server not up after $SERVE_WAIT_MIN min"; tail -n 40 "$LOG"; exit 1
