"""Locating the shift in persona space (confirmatory: fit on discovery data).

PCA is fit ONCE on the baseline role centroids and every condition is projected into that
fixed coordinate system. The Assistant Axis follows Lu et al.:
    axis = mean(default-Assistant activations) - mean(role activations)
computed here on the baseline condition of this experiment's 24 roles (the original used
275 roles; ours is the same contrast on a smaller role set).
"""

from __future__ import annotations

import numpy as np

from .stats import cos


def pca_fit(B: np.ndarray) -> dict[str, np.ndarray]:
    """B [R, H] baseline centroids -> center, components [k, H], explained variance ratio."""
    center = B.mean(0)
    X = B - center
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    var = s**2
    k = int((s > s.max() * 1e-10).sum()) if s.size else 0
    return {"center": center, "components": Vt[:k], "evr": var[:k] / var.sum() if var.sum() > 0 else var[:k],
            "singular_values": s[:k]}


def project(pca: dict[str, np.ndarray], X: np.ndarray, k: int | None = None) -> np.ndarray:
    V = pca["components"] if k is None else pca["components"][:k]
    return (X - pca["center"]) @ V.T


def captured_fraction(pca: dict[str, np.ndarray], v: np.ndarray, ks: list[int]) -> dict[int, float]:
    """||P_k v||^2 / ||v||^2 for the top-k baseline persona PCs (direction, so no centring)."""
    V = pca["components"]
    vn2 = float(v @ v)
    out = {}
    for k in ks:
        kk = min(k, len(V))
        out[k] = float(((V[:kk] @ v) ** 2).sum() / vn2) if vn2 > 0 else float("nan")
    return out


def pairwise_distances(X: np.ndarray) -> np.ndarray:
    sq = (X**2).sum(1)
    d2 = np.clip(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0, None)
    return np.sqrt(d2)


def distance_change(mu: np.ndarray, ib: int, ic: int) -> dict[str, float]:
    """Do personas keep their separation or converge under condition c (vs baseline b)?

    Note: adding the same vector to every persona leaves all distances unchanged, so any
    change here comes from the role-specific (residual) part of the shift.
    """
    Db = pairwise_distances(mu[:, ib])
    Dc = pairwise_distances(mu[:, ic])
    iu = np.triu_indices(len(Db), 1)
    b, c = Db[iu], Dc[iu]
    return {
        "mean_dist_baseline": float(b.mean()),
        "mean_dist_condition": float(c.mean()),
        "mean_dist_ratio": float(c.mean() / b.mean()) if b.mean() > 0 else float("nan"),
        "frac_pairs_closer": float((c < b).mean()),
        "dist_corr": float(np.corrcoef(b, c)[0, 1]) if len(b) > 2 else float("nan"),
        "rms_spread_baseline": float(np.sqrt(((mu[:, ib] - mu[:, ib].mean(0)) ** 2).sum(1).mean())),
        "rms_spread_condition": float(np.sqrt(((mu[:, ic] - mu[:, ic].mean(0)) ** 2).sum(1).mean())),
    }


def assistant_axis(mu_roles_base: np.ndarray, mu_default_base: np.ndarray) -> np.ndarray:
    return mu_default_base - mu_roles_base.mean(0)


def persona_space_report(mu: np.ndarray, conds: list[str], g: dict[str, np.ndarray],
                         mu_default: np.ndarray | None, ks: list[int], base: str = "baseline",
                         fit_rows: list[int] | None = None) -> dict:
    """mu [R, C, H] role centroids (all roles); g = {contrast name: shared direction};
    mu_default [C, H] default-Assistant centroids (or None). PCA and the Assistant contrast
    are fit on `fit_rows` (discovery roles) only; every role and condition is projected."""
    ib = conds.index(base)
    fit = list(range(mu.shape[0])) if fit_rows is None else list(fit_rows)
    pca = pca_fit(mu[fit, ib])
    rep: dict = {"pca_evr": pca["evr"][:10].tolist(), "n_components": int(len(pca["components"]))}
    proj = {c: project(pca, mu[:, i], 3).tolist() for i, c in enumerate(conds)}
    rep["projections_top3"] = proj
    axis = None
    if mu_default is not None and np.all(np.isfinite(mu_default[ib])):
        axis = assistant_axis(mu[fit, ib], mu_default[ib])
        rep["assistant_axis_norm"] = float(np.linalg.norm(axis))
        rep["cos_assistant_axis_pc1"] = cos(axis, pca["components"][0]) if len(pca["components"]) else float("nan")
        rep["default_projections_top3"] = {c: project(pca, mu_default[i][None], 3)[0].tolist()
                                           for i, c in enumerate(conds) if np.all(np.isfinite(mu_default[i]))}
    for name, v in g.items():
        r = {"captured_frac": captured_fraction(pca, v, ks),
             "random_direction_expectation": {k: min(k, len(pca["components"])) / v.size for k in ks},
             "cos_pc": [cos(v, pc) for pc in pca["components"][:5]]}
        if axis is not None:
            r["cos_assistant_axis"] = cos(v, axis)
            # how far the shift moves personas along the assistant axis, relative to the
            # spread of baseline personas along that axis
            ah = axis / np.linalg.norm(axis)
            spread = float((mu[fit, ib] @ ah).std())
            r["shift_along_axis_over_role_sd"] = float(v @ ah / spread) if spread > 0 else float("nan")
        rep[name] = r
    for i, c in enumerate(conds):
        if c != base:
            rep[f"distances_{c}_vs_{base}"] = distance_change(mu[fit], ib, i)
    rep["_pca"] = pca          # stripped before JSON serialisation
    rep["_axis"] = axis
    return rep
