#!/usr/bin/env python3
"""Fail-fast pod preflight (main venv), run before any GPU work by scripts/pod/_lib.sh:
GPUs actually usable, disk, HF token + pinned model revisions reachable, the results
dataset writable, judge API keys live (one tiny call), self-stop configured.

    python scripts/pod/doctor.py --models qwen3.6-27b --need-gb 120 [--judge] [--shutdown stop]

Exit 0 = safe to leave running; otherwise every failed check is listed.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--need-gb", type=float, default=120.0, help="free disk needed under HF_HOME's filesystem")
    p.add_argument("--judge", action="store_true", help="also check the role-adherence judge API key with a live call")
    p.add_argument("--shutdown", default=os.environ.get("SHUTDOWN", ""), help="stop|terminate: check self-stop config")
    p.add_argument("--pod-id", default=os.environ.get("RUNPOD_POD_ID", ""))
    args = p.parse_args()
    ok, bad = [], []

    def check(name, fn):
        try:
            msg = fn()
            ok.append(f"{name}: {msg}")
        except Exception as e:  # report every failure, not just the first
            bad.append(f"{name}: {e}")

    def gpus():
        import torch

        n = torch.cuda.device_count()
        assert n > 0, "no CUDA device visible"
        out = []
        for i in range(n):
            x = torch.ones(16, device=f"cuda:{i}")
            assert (x @ x).item() == 16
            out.append(f"{torch.cuda.get_device_name(i)} {torch.cuda.get_device_properties(i).total_memory / 2**30:.0f}GB")
        return f"{n} x " + ", ".join(sorted(set(out))) + f" (torch {torch.__version__}, cuda {torch.version.cuda})"

    def disk():
        hf = Path(os.environ.get("HF_HOME", "/workspace/hf"))
        hf.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(hf).free / 2**30
        assert free >= args.need_gb, (f"{free:.0f} GB free under {hf}, need {args.need_gb:.0f} "
                                      "(rp up ... --volume-size 200; weights alone are ~55 GB per model)")
        return f"{free:.0f} GB free under {hf}"

    def hf_models():
        from huggingface_hub import HfApi

        from gshift.config import get_model

        api = HfApi()
        out = []
        for key in args.models:
            spec = get_model(key)
            info = api.model_info(spec.hf_id, revision=spec.revision)
            assert info.sha == spec.revision, f"{spec.hf_id}: hub resolved {info.sha}, pinned {spec.revision}"
            out.append(f"{spec.hf_id}@{spec.revision[:8]}")
        return ", ".join(out)

    def hf_dataset():
        repo = os.environ.get("HF_DATASET_REPO")
        if not repo:
            return "WARNING HF_DATASET_REPO unset: results stay on this pod only (no uploads; terminate -> stop)"
        from huggingface_hub import HfApi

        who = HfApi().whoami()
        return f"{repo} (token of {who.get('name')})"

    def judge():
        import asyncio

        from gshift.config import load_experiment
        from gshift.judge import make_client

        cfg = load_experiment()["judge"]
        client = make_client(cfg)                   # raises if the provider's key is unset
        reply = asyncio.run(client("Reply with the single digit 3."))
        assert reply and reply.strip(), f"empty reply from {cfg['provider']}:{cfg['model']}"
        route = f" via {cfg.get('route')}" if cfg.get("route") else ""
        return f"{cfg['provider']}:{cfg['model']}{route} answered {reply.strip()[:20]!r}"

    def self_stop():
        if not args.shutdown:
            return "SHUTDOWN unset: the pod keeps billing after the job (rp down it yourself)"
        assert args.shutdown in ("stop", "terminate"), f"SHUTDOWN={args.shutdown!r}"
        assert os.environ.get("RUNPOD_API_KEY"), "RUNPOD_API_KEY unset (.env): self-stop would be a no-op"
        pid = args.pod_id
        if not pid and os.path.exists("/proc/1/environ"):
            env = Path("/proc/1/environ").read_bytes().split(b"\0")
            pid = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"RUNPOD_POD_ID=")), "")
        assert pid, "RUNPOD_POD_ID not found (pass rp run --env RUNPOD_POD_ID=<id>)"
        return f"SHUTDOWN={args.shutdown} via REST API for pod {pid}"

    check("gpus", gpus)
    check("disk", disk)
    check("hf models", hf_models)
    check("hf dataset", hf_dataset)
    if args.judge:
        check("judge", judge)
    check("self-stop", self_stop)
    for line in ok:
        print("  ok  ", line)
    for line in bad:
        print("  FAIL", line)
    if bad:
        print(f"!! {len(bad)} preflight check(s) failed")
        return 1
    print("ALL PREFLIGHT CHECKS PASSED - safe to leave it running")
    return 0


if __name__ == "__main__":
    sys.exit(main())
