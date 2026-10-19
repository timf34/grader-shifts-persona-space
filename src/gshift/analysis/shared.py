"""Stability / generalisation helpers on ContrastData (see contrast.py)."""

from __future__ import annotations

import itertools

import numpy as np

from .contrast import ContrastData, disp, permutation_null
from .stats import cos


def alignment_null(cd: ContrastData, g_ref: np.ndarray, n_perm: int, seed: int = 0) -> dict:
    """Null for mean per-role cos(Delta_r, g_ref), g_ref estimated on independent data."""
    obs = float(np.nanmean([cos(d, g_ref) for d in disp(cd)]))
    null = permutation_null(cd, n_perm, seed=seed,
                            stat_fn=lambda D: {"m": float(np.nanmean([cos(d, g_ref) for d in D]))})
    xs = np.array([d["m"] for d in null])
    return {"observed": obs, "null_mean": float(xs.mean()), "null_q95": float(np.quantile(xs, 0.95)),
            "p_greater": float((1 + (xs >= obs).sum()) / (1 + len(xs)))}


def split_half_stability(cd: ContrastData, n_splits: int, seed: int = 0, balance_key: str = "group") -> dict:
    """g from two balanced halves of the questions (each half balanced on `balance_key`:
    'group' -> half the original + half the checkable questions; 'family' -> half of each
    family), roles and prompt variants fixed; cos(g_half1, g_half2) over random splits."""
    rng = np.random.default_rng(seed)
    uq = cd.unit_question()
    strata: dict[str, list[str]] = {}
    for q in sorted(set(uq)):
        strata.setdefault(str(cd.qmeta.get(q, {}).get(balance_key, "?")), []).append(q)
    vals = []
    for _ in range(n_splits):
        half = set()
        for members in strata.values():
            perm = rng.permutation(len(members))
            half |= {members[i] for i in perm[: len(members) // 2]}
        inh = np.array([q in half for q in uq])
        if inh.all() or not inh.any():
            continue
        vals.append(cos(disp(cd.subset_units(inh)).mean(0), disp(cd.subset_units(~inh)).mean(0)))
    v = np.array(vals, dtype=float)
    return {"mean_cos": float(np.nanmean(v)) if len(v) else None, "min_cos": float(np.nanmin(v)) if len(v) else None,
            "n_splits": int(len(v)), "balance_key": balance_key,
            "strata_sizes": {k: len(m) for k, m in strata.items()}}


def family_leave_one_out(cd: ContrastData) -> dict:
    fam = cd.unit_attr("family")
    out = {}
    for f in sorted(set(fam)):
        inside = fam == f
        if inside.all():
            continue
        D_in, D_out = disp(cd.subset_units(inside)), disp(cd.subset_units(~inside))
        g_out = D_out.mean(0)
        out[f] = {"mean_role_cos_with_g_other": float(np.nanmean([cos(d, g_out) for d in D_in])),
                  "cos_g_family_g_other": cos(D_in.mean(0), g_out)}
    vals = [v["cos_g_family_g_other"] for v in out.values()]
    return {"families": out, "mean_cos_g_family_g_other": float(np.nanmean(vals)) if vals else None}


def wording_stability(cd: ContrastData) -> dict:
    """Directions estimated separately per discovery wording template (disjoint blocks)."""
    if cd.wording is None:
        return {"skipped": "no wording labels"}
    ws = sorted(w for w in set(cd.wording.tolist()) if w not in ("none", "?"))
    if len(ws) < 2:
        return {"skipped": "fewer than two wordings"}
    g = disp(cd).mean(0)
    gw = {w: disp(cd.subset_units(cd.wording == w)).mean(0) for w in ws}
    return {"cos_with_pooled_g": {w: cos(v, g) for w, v in gw.items()},
            "pairwise_cos": {f"{a}|{b}": cos(gw[a], gw[b]) for a, b in itertools.combinations(ws, 2)},
            "norm_ratio_to_pooled": {w: float(np.linalg.norm(v) / np.linalg.norm(g)) if np.linalg.norm(g) > 0 else None
                                     for w, v in gw.items()}}


def crossfit_dz(cd: ContrastData) -> dict:
    """Leave-one-role-out paired effect size: for each role, project its per-unit contrast
    vectors on g_hat estimated from the OTHER roles and report mean / sd (Cohen's d_z).
    Unit-free, so it is comparable across models; cross-fitting avoids selection bias."""
    D = disp(cd)
    ds, projs = [], []
    for r in range(len(cd.roles)):
        g = np.delete(D, r, 0).mean(0)
        gn = np.linalg.norm(g)
        m = cd.M[r]
        if gn == 0 or m.sum() < 3:
            continue
        x = cd.E[r, m].astype(np.float64) @ (g / gn)
        sd = float(x.std(ddof=1))
        ds.append(float(x.mean() / sd) if sd > 0 else np.nan)
        projs.append(float(x.mean()))
    return {"mean_dz": float(np.nanmean(ds)) if ds else None, "per_role_dz": ds,
            "mean_projection": float(np.nanmean(projs)) if projs else None}
