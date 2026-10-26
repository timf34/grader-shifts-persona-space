#!/usr/bin/env bash
# PSM experiment, pod warm-up: preflight, weights download || EasySteer venv, Q1 inputs the
# experiment needs (betley_repro for the steering scale). Leaves the pod RUNNING (no SHUTDOWN)
# so run_psm.sh starts warm; a watchdog job (scripts/pod/watchdog.sh) bounds the billing.
#
#   rp run <pod> --job prep -- bash scripts/pod/prep_psm.sh qwen3.6-27b
MODEL="${1:?usage: prep_psm.sh <model-key>}"
# shellcheck source=scripts/pod/_lib.sh
source "$(dirname "$0")/_lib.sh"
pod_init prep "$MODEL"
step "preflight"
doctor "$(need_gb)" || finish 2
step "weights download (background) + EasySteer venv"
start_download
es_install
wait_download
step "Q1 inputs (betley_repro) from $HF_DATASET_REPO"
$PY -u scripts/download_outputs.py --model "$MODEL" --what betley_repro || finish 2
ls -la "outputs/$MODEL/betley_repro" | tail -n 3
finish 0
