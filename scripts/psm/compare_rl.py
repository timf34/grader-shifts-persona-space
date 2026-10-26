#!/usr/bin/env python3
"""Before vs after RLVR on one trajectory (OLMo-3 32B Think DPO -> OLMo-3.1 32B Think final), and the
Qwen3.6 -> 3.8 release comparison, on the PSM confirmation data. Same items, prompts and seeds in both
models, so differences are paired by item (bootstrap over items, 2000 draws).

    python scripts/psm/compare_rl.py  -> docs/psm/compare_rl.json + printed summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from gshift.psm.score import load_scored  # noqa: E402

PAIRS = [("olmo3-32b-dpo", "olmo31-32b-final"), ("qwen3.6-27b", "qwen3.8-27b")]
B = 2000


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [2.5, 97.5])] if len(x) else [None, None]


def rows_of(m):
    rows = load_scored(ROOT / "outputs" / m / "psm" / "gen")
    return [r for r in rows if r["task"] == "fc" or r.get("split") == "confirm"]


def item_means(rows, key, filt):
    acc = defaultdict(list)
    for r in rows:
        if filt(r) and r.get(key) is not None:
            acc[r["item_id"]].append(r[key])
    return {i: float(np.mean(v)) for i, v in acc.items()}


def paired(a: dict, b: dict, seed=0):
    ids = sorted(set(a) & set(b))
    x, y = np.array([a[i] for i in ids]), np.array([b[i] for i in ids])
    bi = np.random.default_rng(seed).integers(0, len(ids), (B, len(ids)))
    return {"a": float(x.mean()), "b": float(y.mean()), "diff": float((y - x).mean()),
            "diff_ci95": ci((y - x)[bi].mean(1)), "n_items": len(ids)}


def did(a1, a0, b1, b0, seed=0):
    """(b1 - b0) - (a1 - a0), items paired across all four cells."""
    ids = sorted(set(a1) & set(a0) & set(b1) & set(b0))
    v = np.array([(b1[i] - b0[i]) - (a1[i] - a0[i]) for i in ids])
    ea, eb = np.array([a1[i] - a0[i] for i in ids]), np.array([b1[i] - b0[i] for i in ids])
    bi = np.random.default_rng(seed).integers(0, len(ids), (B, len(ids)))
    return {"effect_a": float(ea.mean()), "effect_a_ci95": ci(ea[bi].mean(1)),
            "effect_b": float(eb.mean()), "effect_b_ci95": ci(eb[bi].mean(1)),
            "did": float(v.mean()), "did_ci95": ci(v[bi].mean(1)), "n_items": len(ids)}


def persona_spread(rows, ctx, dims=None, split="validation", seed=0):
    """Between-persona SD of dilemma pole-a rates (item-bootstrap), non-default personas."""
    acc = defaultdict(list)
    for r in rows:
        if (r["task"] == "fc" and r["context"] == ctx and r["valid"] and r["split"] == split
                and (dims is None or r["dimension"] in dims)):
            acc[(r["persona"], r["item_id"])].append(r["pole_a"])
    ps = sorted({p for p, _ in acc if p != "default"})
    items = sorted({i for _, i in acc})
    M = np.full((len(ps), len(items)), np.nan)
    for (p, i), v in acc.items():
        if p != "default":
            M[ps.index(p), items.index(i)] = np.mean(v)
    return M, items


def sd_boot(M, seed=0):
    bi = np.random.default_rng(seed).integers(0, M.shape[1], (B, M.shape[1]))
    pt = float(np.nanstd(np.nanmean(M, 1), ddof=1))
    bs = np.array([np.nanstd(np.nanmean(M[:, b], 1), ddof=1) for b in bi])
    return pt, bs


def main():
    out = {}
    for a, b in PAIRS:
        ra, rb = rows_of(a), rows_of(b)
        e = {}
        # 1. reporting task: misreport by pressure x context (all personas pooled, conflict items)
        rep = {}
        for ctx in ("base", "auto", "neutral"):
            for pr in ("mild", "strong", "explicit", "suggest"):
                f = lambda r, c=ctx, p=pr: (r["task"] == "rpt" and r["benchmark"].startswith("rpt_") and r["context"] == c  # noqa: E731
                                            and r["kind"] == "conflict" and r["pressure"] == p and r["valid"])
                rep[f"{ctx}|{pr}"] = paired(item_means(ra, "misreport", f), item_means(rb, "misreport", f))
        e["report_misreport"] = rep
        # 2. grading effect on discretionary ('suggest') falsification, and the change across models
        sug = {}
        for c in ("base", "auto", "neutral"):
            for m_, rr in ((a, ra), (b, rb)):
                sug[(m_, c)] = item_means(rr, "misreport", lambda r, c=c: (r["task"] == "rpt" and r["benchmark"].startswith("rpt_")
                                                                            and r["context"] == c and r["pressure"] == "suggest" and r["valid"]))
        e["suggest_auto_minus_base"] = did(sug[(a, "auto")], sug[(a, "base")], sug[(b, "auto")], sug[(b, "base")])
        e["suggest_auto_minus_neutral"] = did(sug[(a, "auto")], sug[(a, "neutral")], sug[(b, "auto")], sug[(b, "neutral")])
        # 3. stated (questionnaire) vs submitted
        st = {}
        for ctx in ("base", "auto"):
            fq = lambda r, c=ctx: r["task"] == "rptq" and r["context"] == c and r["valid"]  # noqa: E731
            st[ctx] = paired(item_means(ra, "choose_favoured", fq), item_means(rb, "choose_favoured", fq))
        e["stated_choose_favoured"] = st
        # 4. dilemma panel: mean pole-a by dimension (base, all personas) and between-persona spread
        dims = sorted({r["dimension"] for r in ra if r["task"] == "fc"})
        dm = {}
        for d in dims:
            f = lambda r, d=d: r["task"] == "fc" and r["context"] == "base" and r["dimension"] == d and r["valid"]  # noqa: E731
            fd = lambda r, d=d: f(r) and r["persona"] == "default"  # noqa: E731
            dm[d] = {"all_personas": paired(item_means(ra, "pole_a", f), item_means(rb, "pole_a", f)),
                     "default": paired(item_means(ra, "pole_a", fd), item_means(rb, "pole_a", fd))}
        e["dilemma_rates_base"] = dm
        sp = {}
        for ctx in ("base", "auto", "human", "neutral"):
            Ma, ia = persona_spread(ra, ctx)
            Mb, ib = persona_spread(rb, ctx)
            if not ia or ia != ib:
                continue
            pa, ba = sd_boot(Ma)
            pb, bb = sd_boot(Mb)
            sp[ctx] = {"a": pa, "b": pb, "diff": pb - pa, "diff_ci95": ci(bb - ba)}
        e["dilemma_persona_spread_validation"] = sp
        out[f"{a}->{b}"] = e
    od = ROOT / "docs" / "psm"
    (od / "compare_rl.json").write_text(json.dumps(out, indent=1))
    for k, e in out.items():
        print(f"===== {k}  (b − a, item-paired 95% CI)")
        print(" reporting misreport (all personas):")
        for key, v in e["report_misreport"].items():
            if v["a"] > 0.005 or v["b"] > 0.005:
                print(f"   {key:18} {v['a']:.2f} -> {v['b']:.2f}  diff {v['diff']:+.2f} {[round(x, 2) for x in v['diff_ci95']]}")
        for name in ("suggest_auto_minus_base", "suggest_auto_minus_neutral"):
            v = e[name]
            print(f" {name}: a {v['effect_a']:+.2f} {[round(x, 2) for x in v['effect_a_ci95']]}, b {v['effect_b']:+.2f} "
                  f"{[round(x, 2) for x in v['effect_b_ci95']]}, change {v['did']:+.2f} {[round(x, 2) for x in v['did_ci95']]}")
        for ctx, v in e["stated_choose_favoured"].items():
            print(f" stated (questionnaire) {ctx}: {v['a']:.2f} -> {v['b']:.2f} diff {v['diff']:+.2f} {[round(x, 2) for x in v['diff_ci95']]}")
        print(" dilemma pole-a rate at base (all personas | default):")
        for d, v in e["dilemma_rates_base"].items():
            A, D = v["all_personas"], v["default"]
            print(f"   {d:20} {A['a']:.2f}->{A['b']:.2f} ({A['diff']:+.2f} {[round(x, 2) for x in A['diff_ci95']]}) | "
                  f"default {D['a']:.2f}->{D['b']:.2f} ({D['diff']:+.2f} {[round(x, 2) for x in D['diff_ci95']]})")
        for ctx, v in e["dilemma_persona_spread_validation"].items():
            print(f" persona spread {ctx}: {v['a']:.3f} -> {v['b']:.3f} diff {v['diff']:+.3f} {[round(x, 3) for x in v['diff_ci95']]}")


if __name__ == "__main__":
    main()
