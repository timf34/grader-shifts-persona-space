#!/usr/bin/env python3
"""PSM analysis (laptop): pull a model's psm/ outputs from the HF dataset and summarise.

    python scripts/psm/analyze.py --models qwen3.6-27b qwen3.8-27b --stage pilot
    python scripts/psm/analyze.py --models ... --stage confirm --no-download

Writes outputs/psm_analysis/<stage>/<model>.json (+ a compact console summary). Memory: the
activation file is [N prompts, 5 layers, H] float16 (~0.3 GB for confirm); one layer is loaded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gshift.io import model_dir, write_json  # noqa: E402
from gshift.psm import analysis as AN  # noqa: E402
from gshift.psm import plan as P  # noqa: E402
from gshift.psm.score import load_scored, score_row, select_steering  # noqa: E402


def download(model: str, stage: str):
    from huggingface_hub import snapshot_download

    pats = [f"psm/{model}/gen/**", f"psm/{model}/hf/acts_{stage}.npz", f"psm/{model}/hf/trait_*", f"psm/{model}/hf/vignette_feats.npz",
            f"psm/{model}/*.json"]
    snapshot_download(os.environ["HF_DATASET_REPO"], repo_type="dataset", allow_patterns=pats,
                      local_dir=str(ROOT / "outputs" / "_hf_psm"))
    src = ROOT / "outputs" / "_hf_psm" / "psm" / model
    dst = model_dir(model) / "psm"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            dst.unlink()
        else:
            return src
    dst.symlink_to(src.resolve())
    return src


def compact(d, depth=0):
    if isinstance(d, dict):
        return {k: compact(v, depth + 1) for k, v in d.items() if not k.startswith("_")}
    if isinstance(d, float):
        return round(d, 4)
    if isinstance(d, list):
        return [compact(x, depth + 1) for x in d]
    return d


def analyse(model: str, stage: str, cfg: dict) -> dict:
    d = model_dir(model) / "psm"
    personas = P.personas(cfg)
    rows = load_scored(d / "gen")
    out: dict = {"model": model, "stage": stage, "n_rows": len(rows)}
    st = cfg[stage]["track_a"]
    ctxs = st["contexts"]
    rpt = [r for r in rows if r["task"] == "rpt" and r["benchmark"].startswith("rpt_")]
    out["track_a_outcomes"] = AN.outcome_table(rpt, personas, ctxs)
    out["track_a_by_pressure"] = AN.by_pressure(rpt, ctxs)
    ta = AN.track_a(rpt, personas, ctxs)
    out["track_a"] = ta
    q = [r for r in rows if r["task"] == "rptq"]
    if q:
        qctx = st.get("questionnaire_contexts", ctxs)
        out["track_a_questionnaire"] = AN.track_a(q, personas, qctx, value="choose_favoured")
        # stated vs submitted, per persona x context
        sv = {}
        for c in qctx:
            if c in ta["_mats"] and c in out["track_a_questionnaire"]["_mats"]:
                a = AN._nanmean(ta["_mats"][c], 1)
                b = AN._nanmean(out["track_a_questionnaire"]["_mats"][c], 1)
                from scipy.stats import spearmanr
                sv[c] = {"spearman_stated_vs_submitted": float(spearmanr(a, b).correlation),
                         "mean_stated": float(np.nanmean(b)), "mean_submitted": float(np.nanmean(a))}
        out["stated_vs_submitted"] = sv
    fc = [r for r in rows if r["task"] == "fc"]
    if fc:
        from gshift.steer.persona_choice import choice_summary
        crow = [{**r, "format": "constrained", "condition": "none", "parsed": r["valid"], "has_answer": True}
                for r in fc]
        out["forced_choice"] = choice_summary(crow, n_boot=1000, n_perm=500)
    acts = d / "hf" / f"acts_{stage}.npz"
    if acts.exists():
        for L in (24, 36, 48):
            out[f"landscape_L{L}"] = AN.landscape(acts, L, personas, ctxs, ta["_mats"])
    # Track B
    steer = {}
    for r in rows:
        if r["benchmark"].startswith("steer_"):
            steer.setdefault((r["context"], r["condition"]), []).append(r)
    if stage == "pilot" and (d / "gen" / "steer_base").exists():
        out["steer_select"] = select_steering(d / "gen" / "steer_base", cfg)
    if stage == "confirm" and steer:
        choice = json.loads((d / "steer_choice.json").read_text())
        out["steer_choice"] = choice
        out["track_b"] = AN.track_b(steer, choice["choice"], cfg["confirm"]["track_b"]["contexts"],
                                    int(cfg["confirm"]["track_b"]["random"]["n"]))
    tz = d / "hf" / "trait_dirs_ending.npz"
    if acts.exists() and tz.exists():
        for L in (32, 36, 40):
            out[f"projection_L{L}"] = AN.projections(acts, tz, L, personas, ctxs, ta["_mats"])
    vf = d / "hf" / "vignette_feats.npz"
    if vf.exists():
        out["trait_negation_check"] = negation_check(vf, cfg)
    ts = d / "hf" / "trait_stats.json"
    if ts.exists():
        out["trait_stats"] = json.loads(ts.read_text())
    tf = d / f"timing_{stage}.json"
    if tf.exists():
        t = json.loads(tf.read_text())
        out["throughput"] = {"generations": sum(x["n"] for x in t), "gen_tokens": sum(x["gen_tokens"] for x in t)}
    return out


def negation_check(vf: Path, cfg: dict) -> dict:
    """Robustness: the direction rebuilt from negation-BALANCED train pairs (both or neither
    ending contains a negation word), its cosine with the main direction, and held-out accuracy."""
    import re
    from gshift.psm.vignettes import load_vignettes, trait_directions
    neg = re.compile(r"\b(no|not|never|n't|nothing|none)\b", re.I)
    vig = {v["id"]: v for v in load_vignettes(cfg["vignettes_file"])}
    z = np.load(vf)
    ids = [str(i) for i in z["ids"]]
    bal = np.array([bool(neg.search(vig[i]["accurate"])) == bool(neg.search(vig[i]["misrepresent"])) for i in ids])
    splits = [str(s) for s in z["splits"]]
    layers = [int(x) for x in z["layers"]]
    feats = {k: z[k] for k in z.files}
    full = trait_directions(feats, splits, layers)
    sub_splits = [s if (s == "heldout" or b) else "excluded" for s, b in zip(splits, bal)]
    sub = trait_directions(feats, sub_splits, layers)
    return {"n_balanced_train": int(sum(s == "train" for s in sub_splits)),
            "by_layer": {str(L): {"cos_with_main": float(full["dirs"][L] @ sub["dirs"][L]),
                                  "heldout_sign_acc_balanced_dir": sub["stats"][L]["heldout_pair_sign_acc"]}
                         for L in layers}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--stage", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    cfg = P.load_cfg()
    od = ROOT / "outputs" / "psm_analysis" / a.stage
    od.mkdir(parents=True, exist_ok=True)
    for m in a.models:
        if not a.no_download:
            download(m, a.stage)
        res = analyse(m, a.stage, cfg)
        write_json(od / f"{m}.json", compact(res))
        print(f"[analyze] {m} {a.stage}: {res['n_rows']} rows -> {od / (m + '.json')}", flush=True)


if __name__ == "__main__":
    main()
