#!/usr/bin/env bash
# EasySteer venv for generation (Q1, plain vLLM) and steering (Q2). Idempotent.
#
# Same pairing as timf34/AttractorStatePrefillAttack capped/easysteer/install.sh (verified
# 2026-08/09): vllm==0.26.0 + the ZJU-REAL/EasySteer-vllm-v1 overlay pinned at 6267ca0
# (the vllm/steer_vectors layout that matches 0.26.0). vLLM 0.26.0 is also the version
# Betley et al. verified their Qwen3.6-27B layer-36 capture on.
# Wheels follow the HOST driver: driver CUDA 12.x -> torch from the cu128 index and
# /workspace/es_venv_cu128; otherwise PyPI defaults (CUDA 13) and /workspace/es_venv.
#
#   bash scripts/pod/install_easysteer.sh        # prints ES_INSTALL_DONE and the venv path
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
VLLM_PIN="${VLLM_PIN:-0.26.0}"
ES_REF="${ES_REF:-6267ca0}"
DRV=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
case "${DRV%%.*}" in
  12) TORCH_INDEX="https://download.pytorch.org/whl/cu128"; ES_VENV="${ES_VENV:-/workspace/es_venv_cu128}" ;;
  *)  TORCH_INDEX="";                                          ES_VENV="${ES_VENV:-/workspace/es_venv}" ;;
esac
echo "driver CUDA ${DRV:-?} -> venv $ES_VENV ${TORCH_INDEX:+(torch wheels from $TORCH_INDEX)}"
[ -x "$ES_VENV/bin/python" ] || python3 -m venv "$ES_VENV"
# shellcheck disable=SC1091
source "$ES_VENV/bin/activate"
pip install -q -U pip
# shellcheck disable=SC2086
pip install -q "vllm==$VLLM_PIN" "transformers>=5.5.3,<5.15" gguf openai pyyaml python-dotenv hf_transfer \
  ${TORCH_INDEX:+--extra-index-url $TORCH_INDEX}
ES_DIR=/workspace/EasySteer-vllm-v1
if [ ! -d "$ES_DIR/.git" ]; then
  rm -rf "$ES_DIR"
  git clone -q https://github.com/ZJU-REAL/EasySteer-vllm-v1.git "$ES_DIR"
fi
(cd "$ES_DIR" && git fetch -q origin && git checkout -q "$ES_REF")
echo "EasySteer overlay at $(cd "$ES_DIR" && git log -1 --format='%h %cd')"
VLLM_DIR=$(python -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
rsync -a "$ES_DIR/vllm/" "$VLLM_DIR"/
[ -f "$VLLM_DIR/steer_vectors/algorithms/direct.py" ] || { echo "!! overlay layout not recognised under $VLLM_DIR"; exit 1; }
python - <<'PY'
import torch, vllm
from vllm.steer_vectors.algorithms import get_algorithm
alg = get_algorithm("direct")(normalize=False)
h = torch.zeros(2, 3); v = torch.tensor([1.0, 2.0, 3.0])
assert torch.allclose(alg._transform(h, v), v.expand(2, 3))
torch.zeros(1).cuda()   # actually initialise the driver; is_available() alone can lie
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda ok on", torch.cuda.get_device_name(0),
      "| direct algorithm ok")
PY
echo "$ES_VENV" > /workspace/.es_venv_path
echo ES_INSTALL_DONE "$ES_VENV"
