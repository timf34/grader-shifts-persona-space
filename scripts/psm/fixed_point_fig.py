#!/usr/bin/env python3
"""Fixed-point comparison across contexts + figure: persona arrows (baseline -> graded) in the top-2 baseline
PCs with the fitted contraction point marked (prompt token, layer 36, open-ended questions).

    python scripts/psm/fixed_point_fig.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "psm"))
from fixed_point import K, SRC, load  # noqa: E402

LABEL = {"qwen3.6-27b": "Qwen3.6-27B", "qwen3.8-27b": "Qwen3.8-27B", "olmo3-32b-dpo": "OLMo-3 32B DPO",
         "olmo31-32b-final": "OLMo-3.1 32B final"}
COL = {"auto": "#d1495b", "human": "#2e86ab", "neutral": "#8a8a8a"}


def fit_point(m, cond, rep="pre", L=36, boot=0, seed=0):
    roles, conds, mu_d, mu_v, md_d, md_v = load(m, rep, L)
    b, g = conds.index("baseline"), conds.index(cond)
    X = (mu_d[:, b] + mu_v[:, b]) / 2
    Y = (mu_d[:, g] + mu_v[:, g]) / 2
    m0 = X.mean(0)
    V = np.linalg.svd(X - m0, full_matrices=False)[2][:K]
    x, d = (X - m0) @ V.T, (Y - X) @ V.T
    spread = float(np.linalg.norm(X - m0, axis=1).mean())

    def one(idx):
        xx, dd = x[idx], d[idx]
        xm, dm = xx - xx.mean(0), dd - dd.mean(0)
        kp = -float((xm * dm).sum() / (xm * xm).sum())
        c = dd.mean(0) + kp * xx.mean(0)
        return kp, c / kp

    kp, p = one(np.arange(len(x)))
    bs = []
    rng = np.random.default_rng(seed)
    for _ in range(boot):
        bs.append(one(rng.integers(0, len(x), len(x))))
    dflt = ((md_d[b] + md_v[b]) / 2 - m0) @ V.T
    return {"roles": roles, "x": x, "y": x + d, "kappa": kp, "p": p, "boot": bs, "spread": spread, "default": dflt,
            "default_y": ((md_d[g] + md_v[g]) / 2 - m0) @ V.T}


def main():
    out = {}
    for m in SRC:
        conds = ["auto", "human"] + (["neutral"] if m.startswith("olmo") else [])
        fits = {c: fit_point(m, c, boot=300) for c in conds}
        e = {}
        for c, f in fits.items():
            kb = np.array([k for k, _ in f["boot"]])
            e[c] = {"kappa": f["kappa"], "kappa_ci95": [float(v) for v in np.percentile(kb, [2.5, 97.5])],
                    "point_norm_over_spread": float(np.linalg.norm(f["p"]) / f["spread"])}
        for a in conds:
            for c in conds:
                if a < c:
                    dist = np.linalg.norm(fits[a]["p"] - fits[c]["p"]) / fits[a]["spread"]
                    bd = [np.linalg.norm(pa - pc) / fits[a]["spread"] for (_, pa), (_, pc) in zip(fits[a]["boot"], fits[c]["boot"])]
                    e[f"dist_{a}_{c}"] = {"d": float(dist), "ci95": [float(v) for v in np.percentile(bd, [2.5, 97.5])]}
        out[m] = e
        print(m, json.dumps({k: ({kk: (round(vv, 2) if isinstance(vv, float) else [round(x, 2) for x in vv]) for kk, vv in v.items()})
                             for k, v in e.items()}))
    (ROOT / "docs" / "psm" / "fixed_point" / "fixed_point_compare.json").write_text(json.dumps(out, indent=1))

    # figure: Qwen3.8 auto, OLMo final auto, OLMo final neutral
    panels = [("qwen3.6-27b", "auto"), ("qwen3.8-27b", "auto"), ("olmo31-32b-final", "auto"), ("olmo31-32b-final", "neutral")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8))
    for ax, (m, c) in zip(axes, panels):
        f = fit_point(m, c)
        x, y = f["x"], f["y"]
        for i in range(len(x)):
            ax.annotate("", xy=y[i, :2], xytext=x[i, :2], arrowprops=dict(arrowstyle="->", color="#9aa3b0", lw=0.7))
        ax.scatter(x[:, 0], x[:, 1], s=8, color="#4a4a4a", zorder=3, label="persona, no statement")
        ax.scatter(f["default"][0], f["default"][1], s=60, marker="*", color="#1f4e79", zorder=4, label="default Assistant")
        ax.scatter(f["p"][0], f["p"][1], s=90, marker="X", color=COL[c], zorder=5, label=f"fitted pull point ({c})")
        for i, r in enumerate(f["roles"]):
            if r in ("evaluator", "examiner", "robot", "cyborg", "actor", "infant", "strategist", "poet", "bard"):
                ax.text(x[i, 0], x[i, 1], r, fontsize=7, color="#333")
        ax.set_title(f"{LABEL[m]} · {c}\npersonas close {f['kappa']:.0%} of the gap to the point", fontsize=9)
        ax.set_xlabel("baseline persona PC1")
        ax.axhline(0, color="#ddd", lw=0.5)
        ax.axvline(0, color="#ddd", lw=0.5)
    axes[0].set_ylabel("baseline persona PC2")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Prompt token, layer 36: personas move toward a point (2-D view of a 10-D fit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "psm" / "figures" / "fixed_point.png", dpi=150)
    print("figure -> docs/psm/figures/fixed_point.png")


if __name__ == "__main__":
    main()
