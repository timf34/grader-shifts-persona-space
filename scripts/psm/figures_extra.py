#!/usr/bin/env python3
"""Extra report figures: overlaps by layer, persona targets of the grading shift, discretionary
falsification by persona, and the before/after-RL comparison.

    python scripts/psm/figures_extra.py  -> docs/psm/figures/confirm_5..8_*.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
import sys  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
OD = ROOT / "docs" / "psm" / "figures"
LABEL = {"qwen3.6-27b": "Qwen3.6-27B", "qwen3.8-27b": "Qwen3.8-27B", "olmo3-32b-dpo": "OLMo-3 32B DPO",
         "olmo31-32b-final": "OLMo-3.1 32B final"}
COL = {"qwen3.6-27b": "#7fb3d5", "qwen3.8-27b": "#1f618d", "olmo3-32b-dpo": "#f0b27a", "olmo31-32b-final": "#ba4a00"}
CTX = {"base": "#4a4a4a", "auto": "#d1495b", "human": "#2e86ab", "neutral": "#9a9a9a"}


def fig_overlaps():
    ov = json.loads((ROOT / "docs" / "psm" / "overlaps" / "overlaps.json").read_text())["overlaps"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    panels = [("pre", "cos_A_G", "Assistant axis · grading (prompt token)"),
              ("resp", "cos_A_G", "Assistant axis · grading (answer tokens)"),
              ("pre", "cos_A_H", "Assistant axis · honesty direction"),
              ("pre", "cos_G_H", "grading · honesty direction")]
    for ax, (rep, key, title) in zip(axes, panels):
        for m, d in ov.items():
            Ls = sorted(int(L) for L in d[rep])
            ys = [d[rep][str(L)].get(key) for L in Ls]
            ax.plot(Ls, ys, "-o", ms=3, color=COL[m], label=LABEL[m])
        ax.axhline(0, color="k", lw=0.6)
        ax.axhspan(-0.03, 0.03, color="k", alpha=0.07)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("layer")
    axes[0].set_ylabel("cosine (grey band ≈ random)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OD / "confirm_5_overlaps.png", dpi=150)
    plt.close(fig)


def fig_targets():
    pt = json.loads((ROOT / "docs" / "psm" / "overlaps" / "overlaps.json").read_text())["persona_targets"]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
    for ax, m in zip(axes, LABEL):
        cs = pt[m]["pre"]["auto_minus_baseline"]["cos"]
        top = sorted(cs, key=lambda r: -cs[r])[:10]
        ys = [cs[r] for r in top]
        cols = ["#d1495b" if r in ("evaluator", "examiner", "reviewer", "judge") else
                ("#2e86ab" if r == "DEFAULT_ASSISTANT" else "#8a8a8a") for r in top]
        ax.barh(range(len(top))[::-1], ys, color=cols)
        ax.set_yticks(range(len(top))[::-1])
        ax.set_yticklabels([r.replace("DEFAULT_ASSISTANT", "default Assistant") for r in top], fontsize=8)
        ax.set_title(LABEL[m], fontsize=9)
        ax.set_xlabel("cos(grading shift, persona offset)")
    fig.suptitle("Which personas does the grading shift point toward? (prompt token, layer 36; red = evaluator family)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OD / "confirm_6_targets.png", dpi=150)
    plt.close(fig)


def fig_discretion():
    from gshift.psm.score import load_scored

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=False)
    for ax, m in zip(axes, ("olmo3-32b-dpo", "olmo31-32b-final")):
        rows = [r for r in load_scored(ROOT / "outputs" / m / "psm" / "gen")
                if r["task"] == "rpt" and r["benchmark"].startswith("rpt_") and r.get("split") == "confirm"
                and r["pressure"] == "suggest" and r["valid"]]
        P = defaultdict(lambda: defaultdict(list))
        for r in rows:
            P[r["persona"]][r["context"]].append(r["misreport"])
        ps = sorted(P, key=lambda p: np.mean(P[p]["base"]))
        for k, c in enumerate(("base", "auto", "neutral")):
            ax.plot([np.mean(P[p][c]) for p in ps], np.arange(len(ps)) + (k - 1) * 0.18, "o", ms=4, color=CTX[c], label=c)
        ax.set_yticks(range(len(ps)))
        ax.set_yticklabels([("▶ " if p == "default" else "") + p for p in ps], fontsize=8)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("falsifies the report ('Could you put K in? It's your call')")
        ax.set_title(LABEL[m], fontsize=10)
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("With discretion, persona decides (exploratory 'suggest' items, held-out)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OD / "confirm_7_discretion.png", dpi=150)
    plt.close(fig)


def fig_rl():
    c = json.loads((ROOT / "docs" / "psm" / "compare_rl.json").read_text())["olmo3-32b-dpo->olmo31-32b-final"]
    items = [("explicit order, no statement", c["report_misreport"]["base|explicit"]),
             ("discretionary, no statement", c["report_misreport"]["base|suggest"]),
             ("discretionary, evaluated automatically", c["report_misreport"]["auto|suggest"]),
             ("discretionary, neutral sentence", c["report_misreport"]["neutral|suggest"]),
             ("stated: would report manager's figure", c["stated_choose_favoured"]["base"])]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for i, (lab, v) in enumerate(items):
        y = len(items) - 1 - i
        ax.plot([v["a"], v["b"]], [y, y], color="#999", lw=1)
        ax.plot([v["a"]], [y], "o", color=COL["olmo3-32b-dpo"], ms=7, label="DPO (before RLVR)" if i == 0 else None)
        ax.plot([v["b"]], [y], "o", color=COL["olmo31-32b-final"], ms=7, label="final (after RLVR)" if i == 0 else None)
        ax.text(1.02, y, f"{v['diff']:+.2f} [{v['diff_ci95'][0]:+.2f}, {v['diff_ci95'][1]:+.2f}]", va="center", fontsize=8)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([lab for lab, _ in items][::-1], fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("falsification rate (all 20 personas, held-out items)")
    ax.set_title("OLMo before vs after RLVR (item-paired change, 95% CI)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OD / "confirm_8_rl.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_overlaps, fig_targets, fig_discretion, fig_rl):
        try:
            f()
        except Exception as e:  # noqa: BLE001
            print(f"[figures_extra] {f.__name__} failed: {e}")
    print("[figures_extra] done")
