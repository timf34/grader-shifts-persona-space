"""Three geometric hypotheses about how a context moves personas (two-state contrasts):

  H1 shared translation  : every persona moves by (nearly) the same vector; separation kept
  H2 convergence         : personas move towards their common centre; separation shrinks
  H3 position-dependent  : each persona's deviation from the shared shift is predicted by
                           where it sits at baseline (beyond uniform contraction)

Cross-fitting. Baseline positions b_r and displacements Delta_r = mu_r,A - mu_r,B share the
noisy baseline estimate mu_r,B; regressing one on the other on the same data manufactures
"convergence" (regression to the mean). Every estimate here therefore takes predictors from
one half of the questions and targets from a DISJOINT half, split by whole question
families (balanced within each question group), both orientations, averaged over all such
splits. Separation is measured with the unbiased cross-product
<mu_r^A - mu_s^A, mu_r^B - mu_s^B> of two independent halves.

All inputs are discovery roles x discovery questions; generalisation to held-out roles and
held-out (validation) questions is reported separately. Uncertainty: role bootstrap;
nulls: role-label permutation between predictor and target.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from .contrast import ContrastData, disp, state_position
from .stats import cos


def family_splits(cd: ContrastData, max_splits: int = 9) -> list[np.ndarray]:
    """Unit masks for half A; whole families, half of each part's families per half."""
    fam = cd.unit_attr("family")
    per_part = []
    for p in cd.part_weight:
        fams = sorted(set(fam[cd.part == p]))
        k = len(fams) // 2
        if k == 0:
            return []
        combos = [set(c) for c in itertools.combinations(fams, k)]
        # keep one of each complementary pair
        uniq = []
        for c in combos:
            if not any(set(fams) - c == u for u in uniq):
                uniq.append(c)
        per_part.append(uniq)
    masks = []
    for choice in itertools.product(*per_part):
        famA = set().union(*choice)
        masks.append(np.array([f in famA for f in fam]))
        if len(masks) >= max_splits:
            break
    return masks


def _center(X: np.ndarray) -> np.ndarray:
    return X - X.mean(0, keepdims=True)


def _pairwise_cross(XA: np.ndarray, XB: np.ndarray) -> float:
    """mean_{r<s} <x_r^A - x_s^A, x_r^B - x_s^B> (unbiased squared separation)."""
    R = len(XA)
    G = XA @ XB.T
    d = np.diag(G)
    S = d[:, None] + d[None, :] - G - G.T
    iu = np.triu_indices(R, 1)
    return float(S[iu].mean())


def _fit_linear(Z: np.ndarray, E: np.ndarray, ridge: float) -> np.ndarray:
    lam = ridge * np.trace(Z.T @ Z) / max(Z.shape[1], 1)
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ E)


def _r2(E: np.ndarray, P: np.ndarray) -> float:
    den = (E**2).sum()
    return float(1 - ((E - P) ** 2).sum() / den) if den > 0 else float("nan")


def _loo_r2(b: np.ndarray, e: np.ndarray, ranks: list[int], ridge: float) -> dict[str, float]:
    """Leave-one-role-out prediction of residual displacements e_r from baseline positions.

    Identical estimator to fitting, for every held-out role r, a rank-k PCA on the other
    roles' centred baseline positions and a ridge map from their k scores to their residual
    displacements -- computed from the R x R Gram matrix of b: all leave-one-out PCAs are
    batched eigendecompositions of (R-1) x (R-1) centred Gram matrices, never SVDs of
    (R-1) x H matrices (H = 5120), which made the permutation null take hours."""
    R = len(b)
    out: dict[str, float] = {}
    Gm = b @ b.T                                    # [R, R]
    ss = np.einsum("ij,ij->i", b, b)                 # ||b_r||^2
    be = np.einsum("ij,ij->i", b, e)                 # b_r . e_r
    # isotropic contraction model: beta_{-r} = sum_{s != r} e_s.b_s / sum_{s != r} ||b_s||^2
    den = ss.sum() - ss
    beta = np.where(den > 0, (be.sum() - be) / np.where(den > 0, den, 1), 0.0)
    out["r2_isotropic"] = _r2(e, beta[:, None] * b)
    if not ranks:
        return out
    n = R - 1
    idx = np.array([np.delete(np.arange(R), r) for r in range(R)])        # [R, n] training rows
    Gt = Gm[idx[:, :, None], idx[:, None, :]]                              # [R, n, n]
    rs = Gt.mean(2)                                                        # [R, n]
    tot = rs.mean(1)                                                       # [R]
    Gc = Gt - rs[:, :, None] - rs[:, None, :] + tot[:, None, None]         # centred Gram
    lam, U = np.linalg.eigh(Gc)                                            # ascending
    lam, U = lam[:, ::-1], U[:, :, ::-1]
    gr = Gm[np.arange(R)[:, None], idx]                                    # [R, n]: b_r . b_j
    gx = gr - gr.mean(1, keepdims=True) - rs + tot[:, None]                # (b_r - m) . (b_j - m)
    kmax = min(max(ranks), n)
    Uk = U[:, :, :kmax]                                                    # [R, n, kmax]
    et_mean = (e.sum(0)[None] - e) / n                                     # [R, H]: training-row mean of e
    # U_k^T E_t for every held-out role in ONE matmul: pad U_k to all R rows (zero at row r)
    Upad = np.zeros((R, R, kmax))
    Upad[np.arange(R)[:, None], idx] = Uk
    UtE = (Upad.transpose(0, 2, 1).reshape(R * kmax, R) @ e).reshape(R, kmax, -1)
    UtE -= Uk.sum(1)[:, :, None] * et_mean[:, None, :]                    # U_k^T (E_t - mean)
    proj = np.einsum("rj,rjk->rk", gx, Uk)                                 # (b_r - m) . u_j  (= z_r,j * sqrt(lam_j))
    for k in ranks:
        kk = min(k, n)
        lk = np.clip(lam[:, :kk], 0, None)                                 # [R, kk]
        keep = lk > (lk.max(1, keepdims=True) * 1e-12)
        # ridge on the orthogonal scores Z (Z^T Z = diag(lam)): W = diag(1/(lam + reg)) Z^T (E_t - mean),
        # reg = ridge x mean(lam); prediction z_r W = sum_j (proj_j / (lam_j + reg)) u_j^T (E_t - mean)
        reg = ridge * np.where(keep, lk, 0).sum(1) / np.maximum(keep.sum(1), 1)
        coef = np.where(keep, proj[:, :kk] / (lk + reg[:, None]), 0.0)
        P = et_mean + np.einsum("rj,rjh->rh", coef, UtE[:, :kk])
        out[f"r2_linear_k{k}"] = _r2(e, P)
    return out


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _aggregate(cd: ContrastData, masks: list[np.ndarray], ranks, ridge, axis_fn=None) -> dict[str, float]:
    rows = []
    for m in masks:
        rows.append(_one_split(cd, m, ranks, ridge, axis_fn))
        rows.append(_one_split(cd, ~m, ranks, ridge, axis_fn))
    keys = sorted({k for r in rows for k in r})
    return {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}


def _halves(cd: ContrastData, mA: np.ndarray) -> dict[str, np.ndarray]:
    """Per-role matched-unit means of half A (units in mA) and half B (the rest): the
    reference-state position and the displacement, computed ONCE per split; every
    permutation / bootstrap replicate only re-indexes these [R, H] arrays."""
    cA, cB = cd.subset_units(mA), cd.subset_units(~mA)
    return {"refA": state_position(cA, "B"), "refB": state_position(cB, "B"), "DA": disp(cA), "DB": disp(cB)}


def _swap(h: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The opposite orientation of a split (predictors from B, targets from A)."""
    return {"refA": h["refB"], "refB": h["refA"], "DA": h["DB"], "DB": h["DA"]}


def _split_stats(h: dict[str, np.ndarray], ranks: list[int], ridge: float, axis: np.ndarray | None = None,
                 perm: np.ndarray | None = None, idx: np.ndarray | None = None) -> dict[str, float]:
    """Statistics for predictors from half A, targets from half B (and the separation /
    shared-energy cross-products, which are symmetric). perm: pair each role's baseline
    position with role perm[r]'s displacements (null); idx: resample roles (bootstrap).
    The condition-state position is reference + displacement (two-state contrast)."""
    refA, refB, DA, DB = h["refA"], h["refB"], h["DA"], h["DB"]
    if idx is not None:
        refA, refB, DA, DB = refA[idx], refB[idx], DA[idx], DB[idx]
    if perm is not None:
        DA, DB = DA[perm], DB[perm]
    conA, conB = refA + DA, refB + DB
    b = _center(refA)
    e = _center(DB)
    out: dict[str, float] = {}
    out["beta"] = float((e * b).sum() / (b**2).sum()) if (b**2).sum() > 0 else float("nan")
    sep_ref = _pairwise_cross(_center(refA), _center(refB))
    sep_con = _pairwise_cross(_center(conA), _center(conB))
    out["separation_ref"] = sep_ref
    out["separation_cond"] = sep_con
    out["separation_ratio"] = sep_con / sep_ref if sep_ref > 0 else float("nan")
    R = len(DA)
    tot = float((DA * DB).sum())
    out["shared_energy_frac_cf"] = float(R * DA.mean(0) @ DB.mean(0) / tot) if tot > 0 else float("nan")
    out.update(_loo_r2(b, e, ranks, ridge))
    gA = DA.mean(0)
    gn = np.linalg.norm(gA)
    if gn > 0:
        u = gA / gn
        out["corr_baseline_vs_shift_along_g"] = _corr(b @ u, DB @ u)
    # leading PC of b from its R x R Gram matrix (b is R x H with R << H)
    w, U = np.linalg.eigh(b @ b.T)
    if w[-1] > 0:
        v1 = U[:, -1] @ b / np.sqrt(w[-1])
        out["corr_baseline_pc1_vs_shift_pc1"] = _corr(b @ v1, e @ v1)
    if axis is not None and np.linalg.norm(axis) > 0:
        ah = axis / np.linalg.norm(axis)
        out["corr_baseline_vs_shift_along_assistant"] = _corr(b @ ah, DB @ ah)
    return out


def _one_split(cd: ContrastData, mA: np.ndarray, ranks: list[int], ridge: float,
               axis_fn=None) -> dict[str, float]:
    """Statistics for predictors from half A (units in mA), targets from half B."""
    return _split_stats(_halves(cd, mA), ranks, ridge, None if axis_fn is None else axis_fn(mA))


def geometry_report(cd: ContrastData, cfg: dict, seed: int = 0, default_cd: ContrastData | None = None,
                    heldout_cd: ContrastData | None = None, validation_cd: ContrastData | None = None) -> dict[str, Any]:
    """cd: discovery roles x discovery questions, two-state contrast with positions."""
    if cd.positions is None:
        return {"skipped": "not a two-state contrast (e.g. interaction): geometry undefined"}
    g = cfg.get("geometry", {})
    ranks = [int(k) for k in g.get("pca_ranks", [3, 5]) if int(k) < len(cd.roles) - 2]
    ridge = float(g.get("ridge", 1e-3))
    masks = family_splits(cd)
    if not masks:
        return {"skipped": "fewer than two question families per group: cannot cross-fit"}

    def axis_fn(mA):  # Assistant contrast from half A only (default control - mean role)
        if default_cd is None or default_cd.positions is None:
            return None
        d = state_position(default_cd.subset_units(mA), "B")[0]
        return d - state_position(cd.subset_units(mA), "B").mean(0)

    halves = [_halves(cd, m) for m in masks]
    rows = []
    for m, h in zip(masks, halves):
        rows.append(_split_stats(h, ranks, ridge, axis_fn(m)))
        rows.append(_split_stats(_swap(h), ranks, ridge, axis_fn(~m)))
    keys = sorted({k for r in rows for k in r})
    obs = {k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}
    rng = np.random.default_rng(seed)
    # null: shuffle which role's targets go with which role's predictors (each role keeps
    # its own baseline position; its displacements are another role's)
    n_perm = int(g.get("n_permutations", 1000))
    null_keys = ["beta"] + [k for k in obs if k.startswith("r2_") or k.startswith("corr_")]
    null = {k: [] for k in null_keys}
    R = len(cd.roles)
    for _ in range(n_perm):
        perm = rng.permutation(R)
        r = _split_stats(halves[rng.integers(len(masks))], ranks, ridge, None, perm=perm)
        for k in null_keys:
            if k in r:
                null[k].append(r[k])
    pvals = {}
    for k, xs in null.items():
        xs = np.array(xs, float)
        xs = xs[np.isfinite(xs)]
        if len(xs) and np.isfinite(obs.get(k, np.nan)):
            if k == "beta":       # two-sided
                pvals[k] = float((1 + (np.abs(xs) >= abs(obs[k])).sum()) / (1 + len(xs)))
            elif k.startswith("r2_"):
                pvals[k] = float((1 + (xs >= obs[k]).sum()) / (1 + len(xs)))
            else:
                pvals[k] = float((1 + (np.abs(xs) >= abs(obs[k])).sum()) / (1 + len(xs)))
    # bootstrap over roles
    n_boot = int(g.get("n_bootstrap", 500))
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, R, size=R)
        if len(set(idx.tolist())) < 4:
            continue
        boot.append(_split_stats(halves[rng.integers(len(masks))], ranks, ridge, None, idx=idx))
    ci = {}
    for k in ("beta", "separation_ratio", "shared_energy_frac_cf", *[f"r2_linear_k{k}" for k in ranks], "r2_isotropic"):
        xs = np.array([b.get(k, np.nan) for b in boot], float)
        xs = xs[np.isfinite(xs)]
        if len(xs):
            ci[k] = {"lo": float(np.quantile(xs, 0.025)), "hi": float(np.quantile(xs, 0.975))}
    res: dict[str, Any] = {"observed": obs, "p_values": pvals, "bootstrap_ci": ci, "n_splits": len(masks),
                           "ranks": ranks, "verdict": _verdict(obs, pvals, ci, ranks)}
    res["generalisation"] = _generalisation(cd, ranks, ridge, heldout_cd, validation_cd)
    return res


def _verdict(obs, p, ci, ranks) -> dict[str, str]:
    """Descriptive labels only -- read the numbers, not the labels."""
    sr = ci.get("separation_ratio", {})
    bt = ci.get("beta", {})
    v = {}
    v["H1_shared_translation"] = ("consistent" if sr and sr["lo"] <= 1 <= sr["hi"] and bt and bt["lo"] <= 0 <= bt["hi"]
                                  else "not consistent")
    v["H2_convergence"] = ("supported" if sr and sr["hi"] < 1 and obs.get("beta", 0) < 0 and p.get("beta", 1) < 0.05
                           else ("expansion" if sr and sr["lo"] > 1 else "not supported"))
    best = max((obs.get(f"r2_linear_k{k}", -np.inf), k) for k in ranks) if ranks else (np.nan, None)
    v["H3_position_dependent"] = (
        f"supported (k={best[1]})" if best[1] is not None and p.get(f"r2_linear_k{best[1]}", 1) < 0.05
        and best[0] > obs.get("r2_isotropic", -np.inf) else "not supported")
    return v


def _generalisation(cd, ranks, ridge, heldout_cd, validation_cd) -> dict:
    """Fit on discovery (predictor = all-discovery baseline positions; target = discovery
    displacements from the same questions is NOT allowed -> use family halves), then test."""
    out = {}
    masks = family_splits(cd)
    if not masks:
        return out
    mA = masks[0]
    cA, cB = cd.subset_units(mA), cd.subset_units(~mA)
    mu_ref = state_position(cA, "B").mean(0)
    mu_d = disp(cB).mean(0)
    b = state_position(cA, "B") - mu_ref
    e = disp(cB) - mu_d
    beta = float((e * b).sum() / (b**2).sum()) if (b**2).sum() > 0 else 0.0
    k = ranks[-1] if ranks else 3
    _, _, Vt = np.linalg.svd(b - b.mean(0), full_matrices=False)
    V = Vt[:k]
    W = _fit_linear((b - b.mean(0)) @ V.T, e - e.mean(0), ridge)

    def evaluate(bt, et):
        Pi = beta * bt
        Pl = ((bt - b.mean(0)) @ V.T) @ W + e.mean(0)
        return {"r2_isotropic": _r2(et, Pi), f"r2_linear_k{k}": _r2(et, Pl),
                "beta_refit": float((et * bt).sum() / (bt**2).sum()) if (bt**2).sum() > 0 else None}

    if heldout_cd is not None and heldout_cd.positions is not None:
        hA, hB = heldout_cd.subset_units(mA), heldout_cd.subset_units(~mA)
        out["heldout_roles"] = evaluate(state_position(hA, "B") - mu_ref, disp(hB) - mu_d)
    if validation_cd is not None and validation_cd.positions is not None:
        # predictor: discovery-question baseline positions (half A); target: displacement on
        # the held-out validation questions (disjoint by construction)
        out["validation_questions"] = evaluate(b, disp(validation_cd) - disp(validation_cd).mean(0))
    return out
