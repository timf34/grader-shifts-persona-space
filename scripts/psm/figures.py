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
    """Fig 1: dilemma panel. Each dot = (persona, dimension): x = p_base - p_default(base),
    y = p_ctx - p_default(ctx). Slope < 1 = personas move toward the default Assistant's choices."""
    ms = [m for m in res if "fc_profiles" in res[m]]
    if not ms:
        return
    fig, axes = plt.subplots(1, len(ms), figsize=(3.8 * len(ms), 3.9), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], ms):
        fp = res[m]["fc_profiles"]["rates"]
        ps = list(fp["base"])
        base = np.array([fp["base"][p] for p in ps], float)
        xd = base[1:] - base[0]
        for c in ("auto", "human", "neutral"):
            if c not in fp:
                continue
            M = np.array([fp[c][p] for p in ps], float)
            yd = M[1:] - M[0]
            ok = np.isfinite(xd) & np.isfinite(yd)
            ax.scatter(xd[ok], yd[ok], s=9, alpha=0.55, color=CTX_COL[c], label=c)
            if ok.sum() > 3:
                b = np.polyfit(xd[ok], yd[ok], 1)[0]
                xx = np.linspace(-1, 1, 10)
                ax.plot(xx, b * xx, color=CTX_COL[c], lw=1, label=f"{c} slope {b:.2f}")
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.7)
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1)
        ax.set_title(LABEL.get(m, m), fontsize=10)
        ax.set_xlabel("persona − default, no statement")
        ax.legend(fontsize=6, loc="upper left")
    axes[0][0].set_ylabel("persona − default, with statement")
    fig.suptitle("Dilemma choices: do personas move toward the default Assistant under grading?", fontsize=10)
    fig.tight_layout()
    fig.savefig(od / f"{stage}_1_profiles.png", dpi=150)
    plt.close(fig)


def fig_convergence(res, stage, od):
    """Fig 2: dilemma panel convergence ratio vs base (mean |p - p_default|, validation items, all dims)."""
    ms = [m for m in res if "forced_choice" in res[m]]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, m in enumerate(ms):
        cells = res[m]["forced_choice"]["cells"]["constrained"]
        for k, c in enumerate(("auto", "human", "neutral")):
            e = cells.get(c, {}).get("none", {}).get("convergence", {}).get("base_none", {}).get("validation", {}).get("all_dimensions")
            if not e:
                continue
            v, ci = e["ratio"], e["ratio_ci95"]
            ax.errorbar([i + (k - 1) * 0.22], [v], yerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=CTX_COL[c],
                        label=c if i == 0 else None, capsize=2)
    ax.axhline(1, color="k", lw=0.8, ls="--")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels([LABEL.get(m, m) for m in ms], fontsize=8)
    ax.set_ylabel("persona spread around default\n(context / no statement)")
    ax.set_title("Dilemma panel: < 1 = personas' decisions converge on the default Assistant's", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(od / f"{stage}_2_convergence.png", dpi=150)
    plt.close(fig)


def fig_landscape(res, stage, od, layer=36):
    """Fig 3: dilemma prompts: activation spread ratio vs base and activation-vs-behaviour Mantel rho."""
    key = f"landscape_fc_L{layer}"
    ms = [m for m in res if key in res[m]]
    if not ms:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, m in enumerate(ms):
        L = res[m][key]["contexts"]
        for k, c in enumerate(("base", "auto", "human", "neutral")):
            if c not in L:
                continue
            x = i + (k - 1.5) * 0.15
            mt = L[c].get("mantel_act_vs_behaviour")
            if mt:
                axes[0].plot([x], [mt["rho"]], "o", color=CTX_COL[c], label=c if i == 0 else None)
            axes[1].plot([x], [L[c]["pairwise_ratio_vs_base"]], "o", color=CTX_COL[c])
    for ax, t, yl in ((axes[0], f"persona activation distances vs choice-profile distances (L{layer})", "Mantel Spearman ρ"),
                      (axes[1], f"persona activation spread on dilemma prompts (L{layer})", "pairwise distance / no statement")):
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([LABEL.get(m, m) for m in ms], fontsize=8)
        ax.set_title(t, fontsize=9)
        ax.set_ylabel(yl)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[1].axhline(1, color="k", lw=0.6, ls="--")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(od / f"{stage}_3_landscape.png", dpi=150)
    plt.close(fig)


def fig_steering(res, stage, od):
    """Fig 4: reporting task, default Assistant: misreport under each steering condition by context,
    pre-registered and exploratory settings; hollow markers = condition fails the competence guard."""
    rows = []
    for m in res:
        for key, tag in (("track_b", "pre-registered"), ("track_b_exploratory", "exploratory")):
            if key in res[m]:
                rows.append((m, key, tag))
    if not rows:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 3.9), sharey=True, squeeze=False)
    for ax, (m, key, tag) in zip(axes[0], rows):
        tb = res[m][key]
        L, s = tb["layer"], tb["strength"]
        order = ["none", f"trait@L{L}_+{s:g}", f"trait@L{L}_-{s:g}"] + \
                [f"random{k}@L{L}_{sg}{s:g}" for k in range(3) for sg in "+-"]
        labels = ["none", "trait +", "trait −"] + [f"r{k}{sg}" for k in range(3) for sg in "+−"]
        for k, (c, e) in enumerate(tb["contexts"].items()):
            q = e.get("conditions", {})
            for j, cond in enumerate(order):
                v = q.get(cond)
                if not v or v.get("rate") is None:
                    continue
                ax.plot([j + (k - 1) * 0.22], [v["rate"]], "o", color=CTX_COL.get(c, "k"),
                        mfc=CTX_COL.get(c, "k") if v["competent"] else "white", label=c if j == 0 else None)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labels, rotation=60, fontsize=7)
        ax.set_title(f"{LABEL.get(m, m)}\n{tag}: L{L}, strength {s:g}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0][0].set_ylabel("misreport rate (default Assistant)")
    axes[0][0].legend(fontsize=7)
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
