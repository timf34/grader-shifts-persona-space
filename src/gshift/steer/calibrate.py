"""Strength calibration on separate prompts (pre-registered rule in configs/steering.yaml)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..io import read_jsonl
from .metrics import degenerate
from .tasks import gsm8k_correct


def calib_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    ans = np.mean([bool(r.get("has_answer")) for r in rows])
    rep = np.mean([degenerate(r.get("answer_text", "")) or degenerate(r.get("reasoning_text", "")[-4000:])
                   for r in rows])
    g = [r for r in rows if r["benchmark"] == "calib_gsm8k"]
    acc = np.mean([bool(gsm8k_correct(r.get("answer_text", ""), r["meta"]["answer"])) for r in g]) if g else np.nan
    return {"answer_rate": float(ans), "repetition_rate": float(rep), "gsm8k_accuracy": float(acc), "n": len(rows)}


def choose_strength(results: dict[str, dict[str, float]], cfg: dict[str, Any], direction: str) -> dict[str, Any]:
    """results: condition name -> calib_metrics. Returns the chosen s* and the table."""
    c = cfg["calibration"]
    base = results["none"]
    grid = sorted(float(s) for s in c["strength_grid"])
    passed = []
    for s in grid:
        ok = True
        for sign in ("+", "-"):
            m = results.get(f"{direction}_{sign}{s:g}")
            if m is None:
                ok = False
                break
            drop = base["gsm8k_accuracy"] - m["gsm8k_accuracy"]
            ok &= (m["answer_rate"] >= c["min_answer_rate"] and m["repetition_rate"] <= c["max_repetition_rate"]
                   and (np.isnan(drop) or drop <= c["max_accuracy_drop"]))
        if not ok:
            break
        passed.append(s)
    chosen = passed[-1] if passed else grid[0]
    return {"s_star": chosen, "eval_strengths": sorted({chosen / 2, chosen}), "passed": passed,
            "none_passed": not passed, "direction": direction,
            "note": "s* = largest grid strength with every smaller strength also passing, both signs"}


def load_calibration_rows(path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in read_jsonl(path):
        out.setdefault(r["condition"], []).append(r)
    return out
