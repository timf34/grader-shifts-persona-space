#!/usr/bin/env python3
"""Stage B analysis (PREREG_AB.md B.5, frozen). Discovery by default. The validation split is
analysed only with --split validation, which refuses to run unless this script is committed
and unmodified (its git blob hash is recorded in the output).

    python scripts/followup/b_analyze.py --model qwen3.6-27b --out followup_out/B
    python scripts/followup/b_analyze.py --model qwen3.6-27b --out followup_out/B --split validation
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N_BOOT = 2000
BOOT_SEED = 20261024
CONDS = ["baseline", "auto_correct", "human_correct", "auto_quality", "human_quality", "blind_spot"]
OUTCOMES = ["shortcut_exec", "shortcut_confirmed", "correct", "no_code", "notes_conflict_answer",
            "notes_conflict_reasoning", "truncated"]
CONTRASTS = {   # weights over conditions, applied per (task, seed) unit
    "blind_spot_minus_baseline": {"blind_spot": 1, "baseline": -1},                       # primary
    "auto_correct_minus_baseline": {"auto_correct": 1, "baseline": -1},
    "human_correct_minus_baseline": {"human_correct": 1, "baseline": -1},
    "auto_quality_minus_baseline": {"auto_quality": 1, "baseline": -1},
    "human_quality_minus_baseline": {"human_quality": 1, "baseline": -1},
    "evaluator_main": {"auto_correct": .5, "auto_quality": .5, "human_correct": -.5, "human_quality": -.5},
    "criterion_main": {"auto_correct": .5, "human_correct": .5, "auto_quality": -.5, "human_quality": -.5},
    "evaluator_x_criterion": {"auto_correct": 1, "human_correct": -1, "auto_quality": -1, "human_quality": 1},
    "betley_matched": {"auto_correct": 1, "human_quality": -1},
}


def git_state(path: Path) -> dict:
    rel = str(path.relative_to(ROOT))
    blob = subprocess.check_output(["git", "-C", str(ROOT), "hash-object", rel]).decode().strip()
    committed = subprocess.run(["git", "-C", str(ROOT), "diff", "--quiet", "HEAD", "--", rel]).returncode == 0
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", rel],
                             capture_output=True).returncode == 0
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    return {"script_blob": blob, "committed_unmodified": committed and tracked, "head": head}


def cube(rows: list[dict]):
    tasks = sorted({r["task_id"] for r in rows})
    seeds = sorted({r["seed_idx"] for r in rows})
    X = {o: np.full((len(tasks), len(seeds), len(CONDS)), np.nan) for o in OUTCOMES}
    ti = {t: i for i, t in enumerate(tasks)}
    for r in rows:
        v = {"shortcut_exec": r["shortcut_exec"], "shortcut_confirmed": r["shortcut_confirmed"],
             "correct": r["outcome"] == "CORRECT", "no_code": r["outcome"] == "NO_CODE",
             "notes_conflict_answer": r["notes_conflict_answer"], "notes_conflict_reasoning": r["notes_conflict_reasoning"],
             "truncated": bool(r.get("truncated"))}
        for o in OUTCOMES:
            X[o][ti[r["task_id"]], seeds.index(r["seed_idx"]), CONDS.index(r["condition"])] = float(v[o])
    return tasks, X


def summarise(X: np.ndarray, rng) -> dict:
    """Rates per condition and contrasts, task-cluster bootstrap CIs."""
    T = X.shape[0]
    idx = [rng.integers(0, T, T) for _ in range(N_BOOT)]

    def stat(A):     # A [T, S, C] -> rates [C], contrast values
        rates = np.nanmean(A, axis=(0, 1))
        con = {k: float(np.nanmean(sum(w * A[:, :, CONDS.index(c)] for c, w in ws.items()))) for k, ws in CONTRASTS.items()}
        return rates, con

    r0, c0 = stat(X)
    boots = [stat(X[i]) for i in idx]
    rb = np.array([b[0] for b in boots])
    out = {"rates": {c: {"mean": float(r0[j]), "lo": float(np.nanquantile(rb[:, j], .025)),
                         "hi": float(np.nanquantile(rb[:, j], .975))} for j, c in enumerate(CONDS)}, "contrasts": {}}
    for k in CONTRASTS:
        v = np.array([b[1][k] for b in boots])
        out["contrasts"][k] = {"mean": c0[k], "lo": float(np.nanquantile(v, .025)), "hi": float(np.nanquantile(v, .975))}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3.6-27b")
    p.add_argument("--out", default=str(ROOT / "followup_out" / "B"))
    p.add_argument("--split", choices=["discovery", "validation"], default="discovery")
    args = p.parse_args()
    gs = git_state(Path(__file__).resolve())
    if args.split == "validation" and not gs["committed_unmodified"]:
        sys.exit("refusing to unseal validation: commit b_analyze.py first (analysis choices must be frozen)")
    d = Path(args.out) / args.model
    name = "scores_discovery.jsonl" if args.split == "discovery" else "scores_validation.sealed.jsonl"
    rows = [json.loads(l) for l in open(d / name)]
    tasks, X = cube(rows)
    rng = np.random.default_rng(BOOT_SEED)
    res = {"split": args.split, "n_tasks": len(tasks), "n_records": len(rows), "git": gs,
           "outcomes": {o: summarise(X[o], rng) for o in OUTCOMES},
           "judge": {"n_judged": sum(1 for r in rows if r.get("judged")),
                     "labels_on_shortcuts": _count([r.get("judge_label") for r in rows if r["outcome"] == "SHORTCUT"]),
                     "labels_on_audit_sample": _count([r.get("judge_label") for r in rows if r.get("judge_audit_sample")])},
           "outcome_counts": {c: _count([r["outcome"] for r in rows if r["condition"] == c]) for c in CONDS}}
    pc = res["outcomes"]["shortcut_exec"]["contrasts"]["blind_spot_minus_baseline"]
    res["primary"] = {"contrast": "blind_spot_minus_baseline (shortcut_exec)", **pc,
                      "meets_discovery_criterion": abs(pc["mean"]) >= 0.10 and (pc["lo"] > 0 or pc["hi"] < 0)}
    (d / f"analysis_{args.split}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"split": args.split, "n_tasks": len(tasks), "primary": res["primary"],
                      "shortcut_exec_rates": {c: round(v["mean"], 3) for c, v in res["outcomes"]["shortcut_exec"]["rates"].items()},
                      "outcome_counts": res["outcome_counts"]}, indent=1))


def _count(xs):
    out = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return out


if __name__ == "__main__":
    main()
