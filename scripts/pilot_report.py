#!/usr/bin/env python3
"""Summarise the discovery-only thinking ON / OFF pilot for one model.

Reads outputs_pilot/thinking_on/<model>/ and outputs_pilot/thinking_off/<model>/ (whatever
stages have run: generation is enough for the first sections) and reports, per arm:
  * truncation, answer / reasoning presence, token lengths (-> generation.max_tokens);
  * generation timings and a GPU-hour estimate for the main grid;
  * role adherence by condition and its matched change vs baseline (if judged);
  * displacement reliability at the primary layer (if extracted): |g| / activation norm,
    leave-one-role-out alignment, split halves by question family, prompt-variant
    leave-one-out, per-role split-half reliability;
  * thinking-mode sensitivity: cos(g_on, g_off) per contrast and representation.
Pilot data never feed the confirmatory analysis.

    python scripts/pilot_report.py --model qwen3.6-27b
    python scripts/pilot_report.py --model qwen3.8-27b --root /workspace/repo/outputs_pilot --json
"""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np

from _common import ROOT, load_experiment

from gshift.config import load_questions, role_categories
from gshift.design import build_requests


@contextlib.contextmanager
def outputs_at(path: Path):
    old = os.environ.get("GSHIFT_OUTPUTS")
    os.environ["GSHIFT_OUTPUTS"] = str(path)
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("GSHIFT_OUTPUTS", None)
        else:
            os.environ["GSHIFT_OUTPUTS"] = old


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def generation_section(model: str, exp: dict, on: bool) -> dict:
    from gshift.io import conditions_present, model_dir, read_jsonl, responses_path, roles_present

    rows = []
    for c in conditions_present(model):
        for r in roles_present(model, c):
            rows += read_jsonl(responses_path(model, c, r))
    want = {r.id for r in build_requests(exp, "pilot", enable_thinking=on)}
    have = {r["id"] for r in rows}
    out = {"n_records": len(rows), "n_designed": len(want), "missing": len(want - have),
           "unexpected": len(have - want)}
    if not rows:
        return out
    gen = np.array([len(r["gen_ids"]) for r in rows])
    out.update({
        "max_tokens": sorted({r["max_tokens"] for r in rows}),
        "truncated_rate": float(np.mean([r["truncated"] for r in rows])),
        "has_answer_rate": float(np.mean([r["has_answer"] for r in rows])),
        "think_closed_rate": float(np.mean([r["think_closed"] for r in rows])),
        "config_ids": sorted({r["config_id"] for r in rows}),
        "tokens": {k: {"p50": pct(a, 50), "p90": pct(a, 90), "p99": pct(a, 99), "max": int(a.max())}
                   for k, a in (("generated", gen),
                                ("reasoning", np.array([r["n_reasoning_tokens"] for r in rows])),
                                ("answer", np.array([r["n_answer_tokens"] for r in rows])))},
        "by_group_condition": {},
    })
    for key in sorted({(r["question_group"], r["condition"]) for r in rows}):
        sub = [r for r in rows if (r["question_group"], r["condition"]) == key]
        g = np.array([len(r["gen_ids"]) for r in sub])
        out["by_group_condition"]["|".join(key)] = {
            "n": len(sub), "truncated_rate": float(np.mean([r["truncated"] for r in sub])),
            "has_answer_rate": float(np.mean([r["has_answer"] for r in sub])),
            "generated_p50": pct(g, 50), "generated_p99": pct(g, 99),
            "answer_p50": pct([r["n_answer_tokens"] for r in sub], 50)}
    out["suggested_max_tokens"] = int(np.ceil(np.percentile(gen, 99) * 1.5 / 512) * 512)
    # timings (written by gshift.generate per chunk)
    t = [x for f in sorted((model_dir(model) / "run_config").glob("timings_*.jsonl")) for x in read_jsonl(f)]
    if t:
        sec = sum(x["seconds"] for x in t)
        n = sum(x["n_requests"] for x in t)
        tok = sum(x["gen_tokens"] for x in t)
        n_main = len(build_requests(exp, "main")) + len(build_requests(exp, "wording_validation")) \
            + len(build_requests(exp, "followup"))
        out["timing"] = {"gpu_seconds": sec, "requests": n, "gen_tokens": tok,
                         "gen_tokens_per_gpu_second": tok / sec if sec else None,
                         "est_gpu_hours_main+validation+followup": sec / n * n_main / 3600 if n else None,
                         "note": "linear extrapolation from pilot chunks at the same settings; "
                                 "sum over workers = GPU time, divide by #GPUs for wall clock"}
    return out


def adherence_section(model: str, exp: dict) -> dict:
    from gshift.io import conditions_present, read_jsonl, roles_present, scores_path

    sc = {}
    for c in conditions_present(model):
        for r in roles_present(model, c):
            for x in read_jsonl(scores_path(model, c, r)):
                sc[x["id"]] = x.get("score")
    if not sc:
        return {"skipped": "not judged"}
    import pandas as pd

    reqs = pd.DataFrame([r.to_dict() for r in build_requests(exp, "pilot") if r.role != exp["default_role"]])
    reqs["score"] = pd.to_numeric(reqs["id"].map(sc), errors="coerce")
    out = {"n_scored": int(reqs["score"].notna().sum()), "by_group_condition": {}, "matched_change_vs_baseline": {}}
    for (g, c), s in reqs.groupby(["question_group", "condition"]):
        v = s["score"].dropna()
        out["by_group_condition"][f"{g}|{c}"] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
                                                 "frac_fully_roleplaying": float((v == 3).mean()) if len(v) else None}
    key = ["role", "prompt_variant", "question_id", "sample"]
    piv = reqs.pivot_table(index=key, columns="condition", values="score", aggfunc="first")
    for c in piv.columns:
        if c != "baseline" and "baseline" in piv:
            d = (piv[c] - piv["baseline"]).dropna()
            out["matched_change_vs_baseline"][c] = {"n": int(len(d)), "mean": float(d.mean()) if len(d) else None}
    return out


def reliability_section(model: str, exp: dict, reps: list[str], layer: int) -> tuple[dict, dict]:
    """Returns (report, {(rep, group, contrast): g}) for the cross-arm comparison."""
    from gshift.analysis import stats as S
    from gshift.analysis.contrast import disp, from_cells
    from gshift.analysis.data import Store
    from gshift.analysis.report import _group_cells
    from gshift.analysis.shared import split_half_stability

    try:
        store = Store.load(model)
    except FileNotFoundError:
        return {"skipped": "not extracted"}, {}
    meta = store.meta
    for col, default in (("wording_set", "none"), ("wording_id", "none")):
        if col not in meta:
            meta[col] = default
    if layer not in store.layers:
        return {"skipped": f"layer {layer} not extracted (have {store.layers})"}, {}
    cats = role_categories(exp)
    roles = [r for r in exp["pilot"]["roles"] if r in set(meta["role"])]
    qmeta = {q["id"]: q for q in load_questions(exp)}
    cfg = exp["analysis"]
    out, gs = {}, {}
    for rep in reps:
        if not store.valid(rep).any():
            out[rep] = {"skipped": "no valid rows"}
            continue
        base = meta[(meta["condition"] == "baseline") & meta["role"].isin(roles)]
        base = base[store.valid(rep, base)]
        act = float(store.norms(rep, layer, base).mean()) if len(base) else float("nan")
        out[rep] = {"mean_activation_norm_baseline": act}
        for group in ("original", "checkable"):
            conds = list(exp["condition_sets"][group])
            cells, _ = _group_cells(store, meta, rep, layer, group, roles, conds, cats, qmeta)
            for name, w in exp["contrasts"][group].items():
                cd = from_cells(cells, name, w, group)
                cd = cd.subset_units(cd.M.any(0))                        # pilot questions only
                if not cd.M.any():
                    continue
                D = disp(cd)
                g = D.mean(0)
                gs[(rep, group, name)] = g
                o = S.gram_stats(D @ D.T)
                pv = np.array([u[1] for u in cd.units])
                pv_loo = [S.cos(disp(cd.subset_units(pv == p)).mean(0), disp(cd.subset_units(pv != p)).mean(0))
                          for p in sorted(set(pv)) if (pv != p).any()]
                sh = split_half_stability(cd, int(cfg.get("n_split_halves", 20)), seed=int(cfg["seed"]),
                                          balance_key="family")
                # per-role reliability: Delta_r from two family-balanced question halves
                fam = cd.unit_attr("family")
                qs = cd.unit_question()
                half = set()
                for f in sorted(set(fam)):
                    members = sorted(set(qs[fam == f]))
                    half |= set(members[::2])
                inh = np.array([q in half for q in qs])
                Da, Db = disp(cd.subset_units(inh)), disp(cd.subset_units(~inh))
                out[rep][f"{group}|{name}"] = {
                    "n_roles": len(cd.roles), "n_units": int(cd.M.any(0).sum()),
                    "g_norm_over_act_norm": float(np.linalg.norm(g) / act) if act > 0 else None,
                    "mean_loo_cos": o["mean_loo_cos"], "mean_pairwise_cos": o["mean_pairwise_cos"],
                    "split_half_family_mean_cos": sh["mean_cos"],
                    "prompt_variant_loo_mean_cos": float(np.nanmean(pv_loo)) if pv_loo else None,
                    "per_role_split_half_mean_cos": float(np.nanmean([S.cos(a, b) for a, b in zip(Da, Db)])),
                }
    return out, gs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--root", default=str(ROOT / "outputs_pilot"), help="contains thinking_on/ and thinking_off/")
    p.add_argument("--experiment", default=None)
    p.add_argument("--reps", nargs="+", default=["pre", "resp", "reason"])
    p.add_argument("--layer", type=int, default=None, help="default: extraction.primary_layer")
    p.add_argument("--json", action="store_true", help="also write <root>/<model>_pilot_report.json")
    args = p.parse_args()
    exp = load_experiment(args.experiment)
    layer = args.layer or int(exp["extraction"]["primary_layer"])
    report: dict = {"model": args.model, "layer": layer, "arms": {}}
    gs_by_arm = {}
    for arm, on in (("thinking_on", True), ("thinking_off", False)):
        e = json.loads(json.dumps(exp))
        e["generation"]["enable_thinking"] = on
        with outputs_at(Path(args.root) / arm):
            sec = {"generation": generation_section(args.model, e, on), "adherence": adherence_section(args.model, e)}
            sec["reliability"], gs_by_arm[arm] = reliability_section(
                args.model, e, [r for r in args.reps if on or r != "reason"], layer)
        report["arms"][arm] = sec
    a, b = gs_by_arm.get("thinking_on", {}), gs_by_arm.get("thinking_off", {})
    from gshift.analysis.stats import cos

    report["thinking_sensitivity_cos_g_on_off"] = {"|".join(k): cos(a[k], b[k]) for k in a if k in b}

    for arm, sec in report["arms"].items():
        g = sec["generation"]
        print(f"\n=== {args.model} {arm}: {g['n_records']}/{g['n_designed']} generated "
              f"({g['missing']} missing, {g['unexpected']} unexpected)")
        if g["n_records"]:
            print(f"  max_tokens {g['max_tokens']}  truncated {g['truncated_rate']:.1%}  "
                  f"has_answer {g['has_answer_rate']:.1%}  think_closed {g['think_closed_rate']:.1%}  "
                  f"config {g['config_ids']}")
            for k, v in g["tokens"].items():
                print(f"  {k:9s} tokens p50={v['p50']:.0f} p90={v['p90']:.0f} p99={v['p99']:.0f} max={v['max']}")
            print(f"  suggested max_tokens >= {g['suggested_max_tokens']}")
            if "timing" in g:
                t = g["timing"]
                print(f"  timing: {t['gpu_seconds'] / 3600:.2f} GPU-h for {t['requests']} requests, "
                      f"{t['gen_tokens_per_gpu_second']:.0f} tok/s/GPU -> main+validation+followup ~"
                      f"{t['est_gpu_hours_main+validation+followup']:.0f} GPU-h")
        ad = sec["adherence"]
        if "skipped" not in ad:
            print("  adherence (mean score):", {k: round(v["mean"], 2) for k, v in ad["by_group_condition"].items()
                                                if v["mean"] is not None})
        rel = sec["reliability"]
        for rep, blk in rel.items() if isinstance(rel, dict) and "skipped" not in rel else []:
            for k, v in blk.items():
                if isinstance(v, dict):
                    print(f"  [{rep}] {k:42s} |g|/act={v['g_norm_over_act_norm']:.4f} loo={v['mean_loo_cos']:.2f} "
                          f"split-half={v['split_half_family_mean_cos'] or float('nan'):.2f} "
                          f"pv-loo={v['prompt_variant_loo_mean_cos'] or float('nan'):.2f} "
                          f"role-rel={v['per_role_split_half_mean_cos']:.2f}")
        if isinstance(rel, dict) and "skipped" in rel:
            print(f"  reliability: {rel['skipped']}")
    if report["thinking_sensitivity_cos_g_on_off"]:
        print("\nthinking ON vs OFF, cos(g_on, g_off):")
        for k, v in report["thinking_sensitivity_cos_g_on_off"].items():
            print(f"  {k:50s} {v:.3f}")
    if args.json:
        from gshift.analysis.report import _clean

        path = Path(args.root) / f"{args.model}_pilot_report.json"
        path.write_text(json.dumps(_clean(report), indent=2))
        print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
