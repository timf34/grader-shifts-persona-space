#!/usr/bin/env bash
# EasySteer venv for generation (Q1, plain vLLM) and steering (Q2). Idempotent: run it on
# every pod start (scripts/pod/_lib.sh es_install does), since apt packages and the CUDA
# compat libraries live on the container disk, which a stop wipes.
#
# vllm 0.26.0 + the ZJU-REAL/EasySteer-vllm-v1 overlay pinned at 6267ca0 (the
# vllm/steer_vectors layout that matches 0.26.0; verified pairing from
# timf34/AttractorStatePrefillAttack capped/easysteer; also the vLLM version Betley et al.
# verified their Qwen3.6-27B layer-36 capture on). transformers pinned exactly as in
# requirements/pod.txt (chat templates are rendered in this venv).
#
# Wheels follow the HOST driver (nvidia-smi "CUDA Version"):
#   >= 13 : PyPI vllm==0.26.0 (CUDA 13 build, torch 2.11.0)       -> $ES_VENV_ROOT/es_venv_cu130
#   12.x  : the official vllm-0.26.0+cu129 release wheel with torch 2.11.0+cu129 and a CPU
#           torchcodec, plus NVIDIA's cuda-compat-12-9 libraries on drivers < 575
#                                                                  -> $ES_VENV_ROOT/es_venv_cu129
#   ($ES_VENV_ROOT defaults to /workspace; _lib.sh sets / when /workspace is a FUSE mount)
#           (the PyPI wheel needs libcudart.so.13 even with a cu128 torch index; recipe and
#           failure modes from timf34/tess-antra-project
#           reports/generation_runtime_repair_2026-10-08.md: cu129 wheel + compat, CPU
#           torchcodec (else libnvrtc.so.13 on `from vllm import LLM`), ninja on PATH)
# Writes /workspace/.es_env (ES_VENV, ES_LD_LIBRARY_PATH, ES_CUDA_LINE); the compat library
# path is applied to vLLM processes only (scripts/pod/_lib.sh es_py), never to the HF venv.
#
#   bash scripts/pod/install_easysteer.sh        # prints ES_INSTALL_DONE <venv>
set -euo pipefail
VLLM_PIN="${VLLM_PIN:-0.26.0}"
ES_REF="${ES_REF:-6267ca0}"
# same exact pin as requirements/pod.txt (chat templates are rendered in this venv)
TRANSFORMERS_PIN="${TRANSFORMERS_PIN:-5.14.1}"
COMPAT_VER="12-9_575.57.08-0ubuntu1"

DRV_CUDA=$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | head -1)
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
case "${DRV_CUDA%%.*}" in
  12) LINE=cu129 ;;
  1[3-9]|[2-9][0-9]) LINE=cu130 ;;
  *) echo "!! host driver ${DRV:-?} (CUDA ${DRV_CUDA:-?}) is too old: need CUDA >= 12.x"; exit 2 ;;
esac
ES_ROOT="${ES_VENV_ROOT:-/workspace}"
ES_VENV="${ES_VENV:-${ES_ROOT%/}/es_venv_$LINE}"
echo "driver ${DRV:-?} (CUDA ${DRV_CUDA:-?}) -> $LINE wheels, venv $ES_VENV"

# torchcodec (a vLLM dependency) needs the FFmpeg shared libraries just to import
if ! command -v ffmpeg >/dev/null 2>&1; then
  (apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg >/dev/null) \
    || echo "(apt-get install ffmpeg failed; the import self-test below will tell if it matters)"
fi

[ -x "$ES_VENV/bin/python" ] || python3 -m venv "$ES_VENV"
PIP=("$ES_VENV/bin/python" -m pip)
"${PIP[@]}" install -q -U pip
COMMON=("transformers==$TRANSFORMERS_PIN" gguf ninja openai pyyaml python-dotenv hf_transfer)
if [ "$LINE" = cu129 ]; then
  PYTAG=$("$ES_VENV/bin/python" -c 'import sys; print(f"cp{sys.version_info[0]}{sys.version_info[1]}")')
  "${PIP[@]}" install -q \
    --extra-index-url https://download.pytorch.org/whl/cu129 --extra-index-url https://download.pytorch.org/whl/cpu \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_PIN}/vllm-${VLLM_PIN}%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
    "torch==2.11.0+cu129" "torchaudio==2.11.0+cu129" "torchvision==0.26.0+cu129" "torchcodec==0.14.0+cpu" \
    "${COMMON[@]}"
  echo "python $PYTAG, $("$ES_VENV/bin/python" -m pip show torchcodec 2>/dev/null | grep -i '^version')"
else
  "${PIP[@]}" install -q "vllm==$VLLM_PIN" "${COMMON[@]}"
fi

ES_LD=""
if [ "$LINE" = cu129 ] && [ "${DRV%%.*}" -lt 575 ]; then
  # forward-compatibility libraries (datacenter GPUs: A100 / H100 / H200), generation only
  if [ ! -f /usr/local/cuda-12.9/compat/libcuda.so.1 ]; then
    . /etc/os-release
    DISTRO="ubuntu${VERSION_ID//./}"
    curl -fsSL --retry 3 \
      "https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/x86_64/cuda-compat-${COMPAT_VER}_amd64.deb" \
      -o /tmp/cuda-compat.deb
    dpkg -i /tmp/cuda-compat.deb >/dev/null
  fi
  ES_LD=/usr/local/cuda-12.9/compat
  echo "driver ${DRV} < 575: using CUDA 12.9 forward-compat libraries from $ES_LD (vLLM processes only)"
fi

ES_DIR=/workspace/EasySteer-vllm-v1
if [ ! -d "$ES_DIR/.git" ]; then
  rm -rf "$ES_DIR"
  git clone -q https://github.com/ZJU-REAL/EasySteer-vllm-v1.git "$ES_DIR"
fi
(cd "$ES_DIR" && { git checkout -q "$ES_REF" 2>/dev/null || { git fetch -q origin && git checkout -q "$ES_REF"; }; })
echo "EasySteer overlay at $(cd "$ES_DIR" && git log -1 --format='%h %cd')"
VLLM_DIR=$("$ES_VENV/bin/python" -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('vllm').origin))")
# plain recursive copy: network filesystems (e.g. nfs4 /workspace) refuse to preserve ownership (cp -a)
cp -rf "$ES_DIR/vllm/." "$VLLM_DIR/"
[ -f "$VLLM_DIR/steer_vectors/algorithms/direct.py" ] || { echo "!! overlay layout not recognised under $VLLM_DIR"; exit 1; }

# self-test with exactly the environment generation will use: venv bin on PATH (vLLM
# launches ninja by name), compat libraries, the full LLM entrypoint import, a real GPU op
export PATH="$ES_VENV/bin:$PATH"
[ -n "$ES_LD" ] && export LD_LIBRARY_PATH="$ES_LD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ninja --version >/dev/null
EXPECT_CUDA=$([ "$LINE" = cu129 ] && echo 12.9 || echo 13)
EXPECT_CUDA="$EXPECT_CUDA" TRANSFORMERS_PIN="$TRANSFORMERS_PIN" "$ES_VENV/bin/python" - <<'PY'
import os

import torch
import transformers
import vllm
from vllm import LLM, SamplingParams  # noqa: F401  (exercises media/dependency imports before any weights)
from vllm.steer_vectors.algorithms import get_algorithm

assert transformers.__version__ == os.environ["TRANSFORMERS_PIN"], transformers.__version__
assert str(torch.version.cuda).startswith(os.environ["EXPECT_CUDA"]), torch.version.cuda
alg = get_algorithm("direct")(normalize=False)
h = torch.zeros(2, 3); v = torch.tensor([1.0, 2.0, 3.0])
assert torch.allclose(alg._transform(h, v), v.expand(2, 3))
x = torch.ones(16, device="cuda")
assert (x @ x).item() == 16
torch.cuda.synchronize()
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda, "transformers",
      transformers.__version__, "| GPU op ok on", torch.cuda.get_device_name(0), "| direct algorithm ok")
PY
printf 'ES_VENV=%s\nES_LD_LIBRARY_PATH=%s\nES_CUDA_LINE=%s\n' "$ES_VENV" "$ES_LD" "$LINE" > /workspace/.es_env
echo "$ES_VENV" > /workspace/.es_venv_path
echo ES_INSTALL_DONE "$ES_VENV"
