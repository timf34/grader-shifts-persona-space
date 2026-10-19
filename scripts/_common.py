"""Shared CLI plumbing for scripts/*.py (path setup, .env, smoke-mode grid reduction)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from gshift.config import load_experiment, role_categories  # noqa: E402
from gshift.design import partition_roles  # noqa: E402

from gshift.smoke import smoke_experiment  # noqa: E402


def base_parser(desc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model key from configs/models.yaml")
    p.add_argument("--experiment", default=None, help="experiment.yaml (default configs/experiment.yaml)")
    p.add_argument("--smoke", action="store_true",
                   help="technical smoke test: tiny grid, written to outputs_smoke/ (never mixed with real runs)")
    p.add_argument("--worker", type=int, default=0, help="data-parallel worker index (roles are partitioned)")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--thinking", choices=["on", "off"], default=None,
                   help="pilot arms: override generation.enable_thinking (and max_tokens from the pilot block); "
                        "outputs go to outputs_pilot/thinking_<on|off>/ unless GSHIFT_OUTPUTS is set")
    return p


def experiment_for(args) -> dict:
    exp = load_experiment(args.experiment)
    if getattr(args, "smoke", False):
        exp = smoke_experiment(exp)
        os.environ["GSHIFT_OUTPUTS"] = os.environ.get("GSHIFT_SMOKE_OUTPUTS", str(ROOT / "outputs_smoke"))
    if getattr(args, "thinking", None):
        on = args.thinking == "on"
        exp["generation"]["enable_thinking"] = on
        mt = exp.get("pilot", {}).get("max_tokens", {})
        exp["generation"]["max_tokens"] = int(mt.get(on, mt.get(str(on).lower(), exp["generation"]["max_tokens"])))
        if not getattr(args, "smoke", False):
            os.environ.setdefault("GSHIFT_OUTPUTS", str(ROOT / "outputs_pilot" / f"thinking_{args.thinking}"))
    return exp


def worker_roles(exp: dict, args) -> list[str] | None:
    """Roles handled by this worker (None = all). The default control goes to worker 0.
    Pilot runs (--thinking) partition only the pilot roles, so workers get balanced shares."""
    if args.num_workers <= 1:
        return None
    pool = list(exp["pilot"]["roles"]) if getattr(args, "thinking", None) and not getattr(args, "smoke", False) \
        else list(role_categories(exp))
    roles = partition_roles(pool, args.worker, args.num_workers)
    if args.worker == 0:
        roles.append(exp["default_role"])
    return roles
