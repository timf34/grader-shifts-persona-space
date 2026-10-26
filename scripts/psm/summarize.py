#!/usr/bin/env python3
"""Headline numbers per model from outputs/psm_analysis/<stage>/<model>.json -> markdown on stdout.

    python scripts/psm/summarize.py --stage confirm > docs/psm/headline_confirm.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["qwen3.6-27b", "qwen3.8-27b", "olmo3-32b-dpo", "olmo31-32b-final"]
LABEL = {"qwen3.6-27b": "Qwen3.6", "qwen3.8-27b": "Qwen3.8", "olmo3-32b-dpo": "OLMo DPO", "olmo31-32b-final": "OLMo final"}


def f(x, n=2):
    if x is None:
        return "–"
    if isinstance(x, list):
        return "[" + ", ".join(f(v, n) for v in x) + "]"
    try:
        return f"{float(x):.{n}f}"
    except (TypeError, ValueError):
        return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="confirm")
    a = ap.parse_args()
    res = {m: json.loads((ROOT / "outputs" / "psm_analysis" / a.stage / f"{m}.json").read_text())
           for m in MODELS if (ROOT / "outputs" / "psm_analysis" / a.stage / f"{m}.json").exists()}
    out = []
    # --- Track A: dilemma panel convergence (validation items, all dimensions) ---
    out.append("### Track A — dilemma panel: between-persona separation and convergence (validation items)\n")
    out.append("| model | separation SD at base (null) | auto ratio [CI] | human ratio [CI] | neutral ratio [CI] |")
    out.append("|---|---|---|---|---|")
    for m, r in res.items():
        fc = r.get("forced_choice")
        if not fc:
            continue
        cells = fc["cells"]["constrained"]
        sep = cells["base"]["none"]["separation"]["validation"]
        sds = [v["sd"] for v in sep.values() if v.get("sd") is not None]
        nulls = [v["null_mean_sd"] for v in sep.values() if v.get("null_mean_sd") is not None]
        row = [LABEL[m], f"{sum(sds) / len(sds):.2f} ({sum(nulls) / len(nulls):.2f})"]
        for c in ("auto", "human", "neutral"):
            e = cells.get(c, {}).get("none", {}).get("convergence", {}).get("base_none", {}).get("validation", {}).get("all_dimensions")
            row.append(f"{f(e['ratio'])} {f(e['ratio_ci95'])}" if e else "–")
        out.append("| " + " | ".join(row) + " |")
    out.append("\nPer dimension (auto / base mean |p − p_default| ratio, validation):\n")
    out.append("| model | " + " | ".join(["risk", "self_others", "honesty", "rules_outcomes", "loyalty_fairness", "autonomy_deference"]) + " |")
    out.append("|---" * 7 + "|")
    for m, r in res.items():
        fc = r.get("forced_choice")
        if not fc:
            continue
        conv = fc["cells"]["constrained"].get("auto", {}).get("none", {}).get("convergence", {}).get("base_none", {}).get("validation", {})
        out.append(f"| {LABEL[m]} | " + " | ".join(f"{f(conv.get(d, {}).get('ratio'))} {f(conv.get(d, {}).get('ratio_ci95'))}"
                                                     for d in ["risk", "self_others", "honesty", "rules_outcomes", "loyalty_fairness", "autonomy_deference"]) + " |")
    # landscape on dilemma prompts
    out.append("\n### Track A — landscape (dilemma prompts, layer 36): activation spread vs base, Mantel ρ (p)\n")
    out.append("| model | " + " | ".join(["base", "auto", "human", "neutral"]) + " |")
    out.append("|---" * 5 + "|")
    for m, r in res.items():
        L = r.get("landscape_fc_L36")
        if not L:
            continue
        cells = []
        for c in ("base", "auto", "human", "neutral"):
            e = L["contexts"].get(c)
            if not e:
                cells.append("–")
                continue
            mt = e.get("mantel_act_vs_behaviour") or {}
            cells.append(f"spread {f(e['pairwise_ratio_vs_base'])}; ρ {f(mt.get('rho'))} (p {f(mt.get('perm_p'), 3)})")
        out.append(f"| {LABEL[m]} | " + " | ".join(cells) + " |")
    # reporting task
    out.append("\n### Reporting task (secondary): misreport rate by pressure × context; refusals\n")
    out.append("| model | context | mild | strong | explicit | suggest | between-persona SD (null) |")
    out.append("|---|---|---|---|---|---|---|")
    for m, r in res.items():
        bp = r.get("track_a_by_pressure", {})
        ta = r.get("track_a", {}).get("contexts", {})
        for c, e in bp.items():
            out.append(f"| {LABEL[m]} | {c} | " + " | ".join(f(e.get(p, {}).get("misreport")) for p in ("mild", "strong", "explicit", "suggest"))
                       + f" | {f(ta.get(c, {}).get('sd'))} (null {f(ta.get(c, {}).get('sd_null_mean'))}) |")
    out.append("\nStated (questionnaire) vs submitted (report) misreport, mean over personas:\n")
    for m, r in res.items():
        sv = r.get("stated_vs_submitted") or {}
        out.append(f"* {LABEL[m]}: " + "; ".join(f"{c}: stated {f(e['mean_stated'])} vs submitted {f(e['mean_submitted'])} (ρ {f(e['spearman_stated_vs_submitted'])})"
                                                for c, e in sv.items()))
    # Track B
    for key, title in (("track_b", "pre-registered choice"), ("track_b_exploratory", "EXPLORATORY one-sided choice (PREREG 6.8)")):
        out.append(f"\n### Track B — reporting task, {title}: misreport (default Assistant)\n")
        out.append("| model | L, s | context | none | trait +s (valid/ctrl) | trait −s (valid/ctrl) | range [CI] | random band | competent random band | interaction vs base [CI] |")
        out.append("|---|---|---|---|---|---|---|---|---|---|")
        for m, r in res.items():
            tb = r.get(key)
            if not tb:
                continue
            L, s = tb["layer"], tb["strength"]
            for c, e in tb["contexts"].items():
                p, n = e.get(f"trait@L{L}_+{s:g}", {}), e.get(f"trait@L{L}_-{s:g}", {})
                rg = e.get("range_trait", {})
                it = tb.get("interaction_vs_base", {}).get(c)
                qc = e.get("conditions", {})
                qp, qn = qc.get(f"trait@L{L}_+{s:g}", {}), qc.get(f"trait@L{L}_-{s:g}", {})
                out.append(f"| {LABEL[m]} | L{L}, {s:g} | {c} | {f(e['none']['rate'])} | {f(p.get('rate'))} ({f(qp.get('valid'))}/{f(qp.get('control_acc'))}) | "
                           f"{f(n.get('rate'))} ({f(qn.get('valid'))}/{f(qn.get('control_acc'))}) | "
                           f"{f(rg.get('range'))} {f(rg.get('ci95'))} | {f(e.get('random_band'))} | {f(e.get('competent_random_band'))} | "
                           f"{(f(it['range_diff']) + ' ' + f(it['ci95'])) if it else '–'} |")
    for key, title in (("track_b_fc", "pre-registered choice"), ("track_b_fc_exploratory", "EXPLORATORY")):
        out.append(f"\n### Track B — dilemma items ({title}): pooled pole-a range (+s − −s) over default + 4 a-priori roles\n")
        out.append("| model | dimension | context | range [CI] | random band | interaction vs base [CI] |")
        out.append("|---|---|---|---|---|---|")
        for m, r in res.items():
            tb = r.get(key)
            if not tb:
                continue
            for d, ed in tb["by_dimension"].items():
                for c, e in ed["contexts"].items():
                    rg = e.get("range_trait", {})
                    it = ed.get("interaction_vs_base", {}).get(c)
                    out.append(f"| {LABEL[m]} | {d} | {c} | {f(rg.get('pooled'))} {f(rg.get('pooled_ci95'))} | {f(e.get('random_band'))} | "
                               f"{(f(it['range_diff']) + ' ' + f(it['ci95'])) if it else '–'} |")
    # projections
    out.append("\n### Trait projection before the decision (layer 36, reporting prompts): shift vs base in persona-SD units\n")
    for m, r in res.items():
        pj = r.get("projection_L36")
        if pj:
            out.append(f"* {LABEL[m]}: " + "; ".join(f"{c}: default {f(e['default'])}, mean {f(e['mean_shift_vs_base'])}"
                                                    for c, e in pj["contexts"].items() if c != "base"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
