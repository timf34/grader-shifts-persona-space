#!/usr/bin/env python3
"""Figures for the PSM report from outputs/psm_analysis/<stage>/<model>.json.

    python scripts/psm/figures.py --stage confirm --models qwen3.6-27b qwen3.8-27b olmo3-32b-dpo olmo31-32b-final
-> docs/psm/figures/<stage>_*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LABEL = {"qwen3.6-27b": "Qwen3.6-27B", "qwen3.8-27b": "Qwen3.8-27B", "olmo3-32b-dpo": "OLMo-3 32B DPO",
         "olmo31-32b-final": "OLMo-3.1 32B final (RLVR)"}
CTX_COL = {"base": "#4a4a4a", "auto": "#d1495b", "human": "#2e86ab", "neutral": "#9a9a9a"}


def load(stage, models):
    out = {}
    for m in models:
        p = ROOT / "outputs" / "psm_analysis" / stage / f"{m}.json"
        if p.exists():
            out[m] = json.loads(p.read_text())
    return out


def fig_profiles(res, stage, od):
    """Fig 1: persona misreport rates by context (conflict items)."""
    ms = list(res)
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 6.5), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], ms):
        ta = res[m]["track_a"]
        ps = ta["personas"]
        order = sorted(ps, key=lambda p: ta["contexts"]["base"]["rates"][p]["p"] or 0)
        for k, (c, e) in enumerate(ta["contexts"].items()):
            y = np.arange(len(order)) + (k - 1.5) * 0.15
            x = [e["rates"][p]["p"] for p in order]
            lo = [e["rates"][p]["ci95"][0] for p in order]
            hi = [e["rates"][p]["ci95"][1] for p in order]
            ax.errorbar(x, y, xerr=[np.subtract(x, lo), np.subtract(hi, x)], fmt="o", ms=3.5, lw=0.8,
                        color=CTX_COL.get(c, "k"), label=c)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order, fontsize=8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("misreport rate (conflict items)")
        ax.set_title(LABEL.get(m, m), fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0][0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Persona decisions on the reporting task, by context", fontsize=11)
    fig.tight_layout()
    fig.savefig(od / f"{stage}_1_profiles.png", dpi=150)
    plt.close(fig)


def fig_convergence(res, stage, od):
    """Fig 2: between-persona spread ratio vs base (noise-free cross-sample SD and MAD to default)."""
    ms = list(res)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for j, (key, title) in enumerate([("convergence_xsplit_ratio", "between-persona SD (cross-sample)"),
                                      ("convergence_mad_ratio", "mean |p - p_default|")]):
        ax = axes[j]
        for i, m in enumerate(ms):
            ctx = res[m]["track_a"]["contexts"]
            for k, c in enumerate([c for c in ("auto", "human", "neutral") if c in ctx]):
                v = ctx[c].get(key)
                ci = ctx[c].get(key + "_ci95", [np.nan, np.nan])
                if v is None:
                    continue
                x = i + (k - 1) * 0.2
                ax.errorbar([x], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=CTX_COL[c],
                            label=c if i == 0 else None)
        ax.axhline(1, color="k", lw=0.8, ls="--")
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([LABEL.get(m, m) for m in ms], rotation=20, fontsize=8)
        ax.set_ylabel("ratio vs base (< 1 = personas converge)")
        ax.set_title(title, fontsize=10)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(od / f"{stage}_2_convergence.png", dpi=150)
    plt.close(fig)


def fig_landscape(res, stage, od, layer=36):
    """Fig 3: activation-vs-behaviour Mantel rho by context, and activation spread ratios."""
    ms = [m for m in res if f"landscape_L{layer}" in res[m]]
    if not ms:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for i, m in enumerate(ms):
        L = res[m][f"landscape_L{layer}"]["contexts"]
        for k, c in enumerate(L):
            x = i + (k - 1.5) * 0.15
            mt = L[c].get("mantel_act_vs_behaviour")
            if mt:
                axes[0].plot([x], [mt["rho"]], "o", color=CTX_COL.get(c, "k"), label=c if i == 0 else None)
            axes[1].plot([x], [L[c]["pairwise_ratio_vs_base"]], "o", color=CTX_COL.get(c, "k"))
    for ax, t, yl in ((axes[0], f"activation vs behaviour distances (L{layer})", "Mantel Spearman rho"),
                      (axes[1], f"persona activation spread (L{layer}, assay prompts)", "pairwise distance / base")):
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([LABEL.get(m, m) for m in ms], rotation=20, fontsize=8)
        ax.set_title(t, fontsize=10)
        ax.set_ylabel(yl)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[1].axhline(1, color="k", lw=0.6, ls="--")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(od / f"{stage}_3_landscape.png", dpi=150)
    plt.close(fig)


def fig_steering(res, stage, od):
    """Fig 4: misreport vs signed trait strength, by context; random-direction points."""
    ms = [m for m in res if "track_b" in res[m]]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(4.2 * len(ms), 4), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], ms):
        tb = res[m]["track_b"]
        L, s = tb["layer"], tb["strength"]
        for c, e in tb["contexts"].items():
            xs, ys, lo, hi = [], [], [], []
            for x in (-s, -s / 2, 0, s / 2, s):
                key = "none" if x == 0 else f"trait@L{L}_{'+' if x > 0 else '-'}{abs(x):g}"
                if key == "none":
                    v, ci = e["none"]["rate"], e["none"]["ci95"]
                elif key in e:
                    v = e[key]["rate"]
                    ci = [v - (e[key]["diff_vs_none"] - e[key]["diff_ci95"][0]), v + (e[key]["diff_ci95"][1] - e[key]["diff_vs_none"])]
                else:
                    continue
                xs.append(x); ys.append(v); lo.append(ci[0]); hi.append(ci[1])
            ax.errorbar(xs, ys, yerr=[np.subtract(ys, lo), np.subtract(hi, ys)], fmt="-o", color=CTX_COL.get(c, "k"),
                        label=c, ms=4, lw=1.2)
            for k in range(3):
                for x in (s, -s):
                    key = f"random{k}@L{L}_{'+' if x > 0 else '-'}{abs(x):g}"
                    if key in e:
                        ax.plot([x], [e[key]["rate"]], "x", color=CTX_COL.get(c, "k"), alpha=0.6)
        ax.set_title(f"{LABEL.get(m, m)}\n(trait direction, layer {L}; x = random)", fontsize=9)
        ax.set_xlabel("signed strength (+ = toward 'misrepresent')")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("misreport rate, default Assistant")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(od / f"{stage}_4_steering.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="confirm")
    ap.add_argument("--models", nargs="+", default=list(LABEL))
    a = ap.parse_args()
    res = load(a.stage, a.models)
    od = ROOT / "docs" / "psm" / "figures"
    od.mkdir(parents=True, exist_ok=True)
    for f in (fig_profiles, fig_convergence, fig_landscape, fig_steering):
        try:
            f(res, a.stage, od)
        except Exception as e:  # noqa: BLE001
            print(f"[figures] {f.__name__} failed: {e}")
    print(f"[figures] -> {od}")


if __name__ == "__main__":
    main()
