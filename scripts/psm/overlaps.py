#!/usr/bin/env python3
"""How do the Assistant axis, the honesty direction and the grading direction relate, layer by layer,
and which personas does the grading shift point toward? Pre/post RL (OLMo DPO -> final) and
Qwen3.6 -> 3.8. Uses saved Q1 analysis vectors (open-ended questions) and the PSM trait directions.

    python scripts/psm/overlaps.py   -> docs/psm/overlaps/overlaps.json + printed tables

Directions (all unit vectors, same residual stream / layer):
  A  = Assistant contrast (default Assistant centroid - mean role centroid, baseline)
  G  = grading shift: auto - baseline (all models); OLMo also auto - neutral (grading-specific)
  H  = honesty direction from third-person vignettes (misrepresent - accurate, ending tokens)
Reported: cosines (|cos| of random 5120-d pairs ~ 0.011 sd), the share of each direction inside the
top-k baseline persona PCs (random expectation k / 5120), and cross-checkpoint cosines of the same
direction. Persona targets: cos(G, persona offset_r) with offset_r = mu_r - mean_r mu (baseline).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT.parent / "grader-shifts-persona-space" / "outputs"
SRC = {"qwen3.6-27b": Q / "qwen3.6-27b" / "analysis" / "vectors",
       "qwen3.8-27b": Q / "qwen3.8-27b" / "analysis" / "vectors",
       "olmo3-32b-dpo": ROOT / "outputs" / "_hf_q1" / "olmo3-32b-dpo" / "analysis" / "vectors",
       "olmo31-32b-final": ROOT / "outputs" / "_hf_q1" / "olmo31-32b-final" / "analysis" / "vectors"}
TRAIT = {m: ROOT / "outputs" / m / "psm" / "hf" / "trait_dirs_ending.npz" for m in SRC}
TRAIT["olmo31-32b-final"] = ROOT / "outputs" / "olmo31-32b-final" / "psm" / "hf" / "trait_dirs_ending.npz"
LAYERS = [16, 24, 32, 36, 40, 48]
PAIRS = [("qwen3.6-27b", "qwen3.8-27b"), ("olmo3-32b-dpo", "olmo31-32b-final")]


def unit(v):
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def cos(a, b):
    return float(unit(a) @ unit(b))


def load(m, rep, L, name):
    p = SRC[m] / f"original_{rep}_L{L}_{name}.npy"
    return np.load(p) if p.exists() else None


def centroids(m, rep, L):
    z = np.load(SRC[m] / f"original_{rep}_L{L}_condition_centroids.npz", allow_pickle=True)
    roles = [str(r) for r in z["roles"]]
    conds = [str(c) for c in z["conds"]]
    mu = (z["mu_discovery"] + z["mu_validation"]) / 2          # [R, C, H]
    md = (z["mu_default_discovery"] + z["mu_default_validation"]) / 2
    return roles, conds, mu.astype(np.float64), md.astype(np.float64)


def captured(pcs, v, k):
    V = pcs[:k].astype(np.float64)
    u = unit(v)
    return float(((V @ u) ** 2).sum())


def trait(m, L):
    p = TRAIT[m]
    if not p.exists():
        return None
    z = np.load(p)
    return z[f"L{L}"] if f"L{L}" in z.files else None


def main():
    out = {"overlaps": {}, "cross_checkpoint": {}, "persona_targets": {}}
    for m in SRC:
        for rep in ("pre", "resp"):
            for L in LAYERS:
                A = load(m, rep, L, "assistant_contrast")
                if A is None:
                    continue
                G = load(m, rep, L, "auto_minus_baseline")
                Gn = load(m, rep, L, "auto_minus_neutral")
                N = load(m, rep, L, "neutral_minus_baseline")
                H = trait(m, L)
                pcs = load(m, rep, L, "baseline_pcs")
                e = {"cos_A_G": cos(A, G)}
                if H is not None:
                    e.update({"cos_A_H": cos(A, H), "cos_G_H": cos(G, H)})
                if Gn is not None:
                    e["cos_A_Gspec"] = cos(A, Gn)
                    if H is not None:
                        e["cos_Gspec_H"] = cos(Gn, H)
                if N is not None:
                    e["cos_A_neutral"] = cos(A, N)
                for k in (5, 20):
                    e[f"G_in_top{k}"] = captured(pcs, G, k)
                    if H is not None:
                        e[f"H_in_top{k}"] = captured(pcs, H, k)
                    e[f"random_expect_top{k}"] = k / A.size
                out["overlaps"].setdefault(m, {}).setdefault(rep, {})[str(L)] = e
    for a, b in PAIRS:
        for rep in ("pre", "resp"):
            for L in LAYERS:
                e = {}
                for name in ("assistant_contrast", "auto_minus_baseline", "auto_minus_neutral"):
                    va, vb = load(a, rep, L, name), load(b, rep, L, name)
                    if va is not None and vb is not None:
                        e[name] = cos(va, vb)
                ta, tb = trait(a, L), trait(b, L)
                if ta is not None and tb is not None and rep == "pre":
                    e["honesty"] = cos(ta, tb)
                out["cross_checkpoint"].setdefault(f"{a}->{b}", {}).setdefault(rep, {})[str(L)] = e
    # which personas does the grading shift point toward?
    for m in SRC:
        for rep in ("pre", "resp"):
            L = 36
            roles, conds, mu, md = centroids(m, rep, L)
            b = conds.index("baseline")
            base = mu[:, b]
            center = base.mean(0)
            off = base - center
            res = {}
            for gname in ("auto_minus_baseline", "auto_minus_neutral"):
                G = load(m, rep, L, gname)
                if G is None:
                    continue
                cs = {r: cos(G, off[i]) for i, r in enumerate(roles)}
                cs["DEFAULT_ASSISTANT"] = cos(G, md[b] - center)
                ranked = sorted(cs, key=lambda r: -cs[r])
                # per-persona movement: does each role's own graded shift move it toward the evaluator?
                res[gname] = {"cos": cs, "top": ranked[:8], "bottom": ranked[-5:]}
            # per-persona: fraction of roles whose own shift (auto - baseline) reduces distance to 'evaluator'
            if "auto" in conds and "evaluator" in roles:
                a = conds.index("auto")
                ev = roles.index("evaluator")
                d0 = np.linalg.norm(mu[:, b] - mu[ev, b], axis=1)
                d1 = np.linalg.norm(mu[:, a] - mu[ev, a], axis=1)
                dd0 = np.linalg.norm(mu[:, b] - md[b], axis=1)
                dd1 = np.linalg.norm(mu[:, a] - md[a], axis=1)
                res["distance_change"] = {"to_evaluator_median_ratio": float(np.median(d1[np.arange(len(roles)) != ev] / d0[np.arange(len(roles)) != ev])),
                                          "to_default_median_ratio": float(np.median(dd1 / dd0))}
            out["persona_targets"].setdefault(m, {})[rep] = res
    od = ROOT / "docs" / "psm" / "overlaps"
    od.mkdir(parents=True, exist_ok=True)
    (od / "overlaps.json").write_text(json.dumps(out, indent=1))
    # ---- print ----
    print("== cosines (pre-token | answer-token), by layer; random |cos| ~ 0.01")
    for m, d in out["overlaps"].items():
        print(f"-- {m}")
        for rep in ("pre", "resp"):
            for L, e in d.get(rep, {}).items():
                print(f"  {rep:4} L{L}: A·G {e['cos_A_G']:+.2f}" + (f"  A·Gspec {e['cos_A_Gspec']:+.2f}" if 'cos_A_Gspec' in e else "")
                      + (f"  A·H {e['cos_A_H']:+.2f}  G·H {e['cos_G_H']:+.2f}" if 'cos_A_H' in e else "")
                      + (f"  Gspec·H {e['cos_Gspec_H']:+.2f}" if 'cos_Gspec_H' in e else "")
                      + f"  | G in top20 PCs {e['G_in_top20']:.2f}" + (f"  H in top20 {e['H_in_top20']:.2f}" if 'H_in_top20' in e else "")
                      + f" (random {e['random_expect_top20']:.3f})")
    print("\n== same direction across checkpoints (cos)")
    for k, d in out["cross_checkpoint"].items():
        for rep, dl in d.items():
            print(f"  {k} {rep}: " + "; ".join(f"L{L} " + ", ".join(f"{n[:9]} {v:+.2f}" for n, v in e.items()) for L, e in dl.items() if L in ("24", "36", "48")))
    print("\n== personas the grading shift points toward (layer 36)")
    for m, d in out["persona_targets"].items():
        for rep, r in d.items():
            for g, e in r.items():
                if g == "distance_change":
                    print(f"  {m} {rep}: median distance ratio auto/base: to evaluator {e['to_evaluator_median_ratio']:.3f}, to default {e['to_default_median_ratio']:.3f}")
                    continue
                print(f"  {m} {rep} {g}: top {[(t, round(e['cos'][t], 2)) for t in e['top'][:6]]}; default {e['cos']['DEFAULT_ASSISTANT']:+.2f}; bottom {e['bottom'][-3:]}")


if __name__ == "__main__":
    main()
