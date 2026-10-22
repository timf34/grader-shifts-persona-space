#!/usr/bin/env python3
"""Calibrate the cheap Q2 judges against Betley et al.'s judges on a random subset (CPU + API).

The full stage A set is judged with Claude Haiku 5.5 (configs/steering_stage_a.yaml `judges`).
This re-judges a seeded random subset of `calibration_n` School of Reward Hacks answers and
`calibration_n` agentic-misalignment replies with the calibration models (Sonnet 5 for SoRH,
Sonnet 4.6 for agentic, Betley et al.'s settings) and reports agreement:
  SoRH: Pearson / Spearman of metric, quality and gap; mean difference (cheap - reference)
  agentic: agreement and Cohen's kappa on `harmful`, and on the raw classifier verdict
-> steering/{sorh,agentic}_judgements_calib.jsonl and steering/judge_calibration.json

    python -u scripts/17_judge_calibration.py --model qwen3.6-27b --steering-config configs/steering_stage_a.yaml
"""

import json
import random

import numpy as np
from _common import ROOT, base_parser, experiment_for

from gshift.config import load_yaml
from gshift.io import read_jsonl, write_json
from gshift.steer.vectors import steering_dir


def subset(files, n, seed, keep):
    rows = [r for f in files for r in read_jsonl(f) if keep(r)]
    random.Random(seed).shuffle(rows)
    return rows[:n]


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return po, (po - pe) / (1 - pe) if pe < 1 else float("nan")


def main():
    p = base_parser(__doc__)
    p.add_argument("--steering-config", default=str(ROOT / "configs" / "steering_stage_a.yaml"))
    p.add_argument("--seed", type=int, default=20261022)
    args = p.parse_args()
    experiment_for(args)
    cfg = load_yaml(args.steering_config)
    j = cfg["judges"]
    n = int(j.get("calibration_n", 300))
    sd = steering_dir(args.model)
    tmp = sd / "calibration_subsets"
    tmp.mkdir(exist_ok=True)
    out = {"model": args.model, "n_target": n, "cheap": {"sorh": j["sorh_model"], "agentic": j["agentic_model"]},
           "reference": {"sorh": j["sorh_calibration_model"], "agentic": j["agentic_calibration_model"]}}

    # SoRH (scoring-rule prompts only: both judges apply)
    from gshift.steer.judges import judge_sorh

    rows = subset(sorted((sd / "gen" / "sorh").glob("*.jsonl")), n, args.seed, lambda r: r.get("has_answer"))
    f = tmp / "sorh_subset.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ref_path = sd / "sorh_judgements_calib.jsonl"
    judge_sorh([f], ref_path, {**j, "sorh_model": j["sorh_calibration_model"]})
    cheap = {(r["id"], r["condition"]): r for r in read_jsonl(sd / "sorh_judgements.jsonl")}
    ref = {(r["id"], r["condition"]): r for r in read_jsonl(ref_path)}
    res = {}
    for k in ("metric", "quality", "gap"):
        pairs = [(cheap[key][k], ref[key][k]) for key in ref if key in cheap
                 and cheap[key].get(k) is not None and ref[key].get(k) is not None]
        if len(pairs) > 5:
            a, b = np.array(pairs, float).T
            from scipy.stats import spearmanr

            res[k] = {"n": len(pairs), "pearson": float(np.corrcoef(a, b)[0, 1]),
                      "spearman": float(spearmanr(a, b).statistic), "mean_cheap_minus_ref": float((a - b).mean())}
    out["sorh"] = res

    # agentic misalignment
    from gshift.steer.agentic import judge_agentic

    rows = subset(sorted((sd / "gen" / "agentic").glob("*.jsonl")), n, args.seed, lambda r: r.get("has_answer"))
    f = tmp / "agentic_subset.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ref_path = sd / "agentic_judgements_calib.jsonl"
    judge_agentic([f], ref_path, {**cfg, "judges": {**j, "agentic_model": j["agentic_calibration_model"]}})
    cheap = {(r["id"], r["condition"]): r for r in read_jsonl(sd / "agentic_judgements.jsonl")}
    ref = {(r["id"], r["condition"]): r for r in read_jsonl(ref_path)}
    res = {}
    for k in ("harmful", "classifier_verdict"):
        pairs = [(cheap[key][k], ref[key][k]) for key in ref if key in cheap
                 and isinstance(cheap[key].get(k), bool) and isinstance(ref[key].get(k), bool)]
        if pairs:
            a, b = zip(*pairs)
            po, kp = kappa(a, b)
            res[k] = {"n": len(pairs), "agreement": po, "cohen_kappa": kp,
                      "rate_cheap": float(np.mean(a)), "rate_ref": float(np.mean(b))}
    out["agentic"] = res
    write_json(sd / "judge_calibration.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
