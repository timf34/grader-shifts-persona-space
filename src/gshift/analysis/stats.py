"""Is the shift shared? Statistics of the role-displacement matrix D [R, H], its null and CIs.

Nearly every statistic is a function of the Gram matrix G = D D^T (R x R) and of D v for
a few reference vectors v, so the permutation null and the bootstrap only ever need
G for each resample.

Resampling (sign-flip null, roles x questions bootstrap) lives in contrast.py. In the
bootstrap, pairs of copies of the same role are excluded from pairwise / leave-one-out
statistics via `origin`.
"""

from __future__ import annotations

import numpy as np



# ---------------------------------------------------------------------------
# statistics from the Gram matrix
# ---------------------------------------------------------------------------

def _safe_div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def gram_stats(G: np.ndarray, origin: np.ndarray | None = None, Dv: dict[str, np.ndarray] | None = None,
               vnorm: dict[str, float] | None = None) -> dict[str, float]:
    """Scalar summaries of D from G = D D^T.

    origin[r] = original role index (bootstrap copies share an origin; pairs of copies
    are excluded from pairwise and leave-one-out terms).
    Dv[name] = D @ v for reference vectors v (vnorm[name] = ||v||).
    """
    R = G.shape[0]
    origin = np.arange(R) if origin is None else np.asarray(origin)
    same = origin[:, None] == origin[None, :]
    diag = np.clip(np.diag(G), 0, None)
    norms = np.sqrt(diag)
    tot = G.sum()
    g2 = tot / R**2
    out: dict[str, float] = {
        "g_norm": float(np.sqrt(max(g2, 0))),
        "g_norm_sq": float(g2),
        "mean_role_norm": float(norms.mean()),
        "rms_role_norm": float(np.sqrt(diag.mean())),
    }
    # pairwise cosines between role displacements (distinct origins only)
    C = _safe_div(G, np.outer(norms, norms))
    off = ~same
    out["mean_pairwise_cos"] = float(np.nanmean(C[off])) if off.any() else float("nan")
    out["frac_pairwise_cos_positive"] = float(np.nanmean(C[off] > 0)) if off.any() else float("nan")
    # leave-one-out alignment: cos(Delta_r, mean of roles with a different origin)
    M = off.astype(np.float64)
    k = M.sum(1)
    dot_loo = _safe_div((G * M).sum(1), k)                 # Delta_r . g_{-r}
    sq_loo = _safe_div(np.einsum("ij,ij->i", M @ G, M) , k**2)  # ||g_{-r}||^2
    loo = _safe_div(dot_loo, norms * np.sqrt(np.clip(sq_loo, 0, None)))
    out["mean_loo_cos"] = float(np.nanmean(loo))
    out["min_loo_cos"] = float(np.nanmin(loo))
    # alignment with the full mean (includes self; reported for completeness)
    dot_g = G.sum(1) / R
    out["mean_cos_to_g"] = float(np.nanmean(_safe_div(dot_g, norms * np.sqrt(max(g2, 1e-300)))))
    # spectrum of D (uncentred): how much of the displacement energy is one direction
    ev = np.clip(np.linalg.eigvalsh(G)[::-1], 0, None)
    out["sv1_energy_frac"] = float(ev[0] / ev.sum()) if ev.sum() > 0 else float("nan")
    out["participation_ratio"] = float(ev.sum() ** 2 / (ev**2).sum()) if ev.sum() > 0 else float("nan")
    # residual after removing the shared displacement: 1 - R||g||^2 / sum ||Delta_r||^2
    out["shared_energy_frac"] = float(R * g2 / diag.sum()) if diag.sum() > 0 else float("nan")
    out["residual_energy_frac"] = 1.0 - out["shared_energy_frac"]
    # unit-normalised average (sensitivity): ||mean_r Delta_r/||Delta_r||| and cos with g
    Gn = C
    gu2 = np.nansum(Gn) / R**2
    gdotgu = np.nansum(_safe_div(G, norms[None, :])) / R**2
    out["g_unitavg_norm"] = float(np.sqrt(max(gu2, 0)))
    out["cos_g_unitavg"] = float(gdotgu / np.sqrt(max(g2, 1e-300)) / np.sqrt(max(gu2, 1e-300)))
    for name, d in (Dv or {}).items():
        vn = (vnorm or {}).get(name, 1.0)
        gv = float(d.mean())
        out[f"cos_g_{name}"] = gv / (np.sqrt(max(g2, 1e-300)) * vn)
        out[f"mean_cos_role_{name}"] = float(np.nanmean(_safe_div(d, norms * vn)))
    return out


def per_role_stats(D: np.ndarray, roles: list[str]) -> list[dict]:
    """Per-role magnitudes, LOO alignment and influence on g."""
    R = D.shape[0]
    G = D @ D.T
    norms = np.sqrt(np.clip(np.diag(G), 0, None))
    g = D.mean(0)
    gn = np.linalg.norm(g)
    out = []
    for r in range(R):
        rest = np.delete(D, r, 0).mean(0)
        rn = np.linalg.norm(rest)
        out.append({
            "role": roles[r],
            "norm": float(norms[r]),
            "loo_cos": float(D[r] @ rest / (norms[r] * rn)) if norms[r] > 0 and rn > 0 else float("nan"),
            "cos_to_g": float(D[r] @ g / (norms[r] * gn)) if norms[r] > 0 and gn > 0 else float("nan"),
            # share of g's length contributed along g: sums to 1 over roles
            "proj_share": float((D[r] @ g) / (R * gn**2)) if gn > 0 else float("nan"),
            # how much g turns if this role is dropped
            "cos_g_dropped": float(g @ rest / (gn * rn)) if gn > 0 and rn > 0 else float("nan"),
        })
    return out


def influence_summary(D: np.ndarray, top_k: tuple[int, ...] = (1, 3)) -> dict[str, float]:
    """Is g driven by a few unusually large shifts?"""
    norms = np.linalg.norm(D, axis=1)
    g = D.mean(0)
    gn = np.linalg.norm(g)
    order = np.argsort(-norms)
    shares = (D @ g) / (len(D) * gn**2) if gn > 0 else np.full(len(D), np.nan)
    out = {
        "norm_cv": float(norms.std() / norms.mean()) if norms.mean() > 0 else float("nan"),
        "max_norm_over_median": float(norms.max() / np.median(norms)) if np.median(norms) > 0 else float("nan"),
        "max_proj_share": float(np.nanmax(shares)),
        "top3_proj_share": float(np.sort(shares)[::-1][:3].sum()),
    }
    for k in top_k:
        if k >= len(D):
            continue
        rest = D[order[k:]].mean(0)
        out[f"cos_g_without_top{k}"] = float(rest @ g / (np.linalg.norm(rest) * gn)) if gn > 0 else float("nan")
        out[f"norm_ratio_without_top{k}"] = float(np.linalg.norm(rest) / gn) if gn > 0 else float("nan")
    return out


def spectrum(D: np.ndarray) -> dict[str, list[float]]:
    """Singular values of D (uncentred) and of the residual D - 1 g^T."""
    s = np.linalg.svd(D, compute_uv=False)
    Dr = D - D.mean(0, keepdims=True)
    sr = np.linalg.svd(Dr, compute_uv=False)
    return {"singular_values": s.tolist(), "residual_singular_values": sr.tolist(),
            "energy_frac": (s**2 / (s**2).sum()).tolist() if (s**2).sum() > 0 else [],
            "residual_energy_frac": (sr**2 / (sr**2).sum()).tolist() if (sr**2).sum() > 0 else []}


def residual_category_structure(D: np.ndarray, categories: list[str], n_perm: int = 2000,
                                seed: int = 0) -> dict[str, float]:
    """Do residual displacements (after removing g) cluster by role category?

    Statistic: between-category energy fraction of the residuals; null: shuffle the
    category labels across roles.
    """
    Dr = D - D.mean(0, keepdims=True)
    labels = np.asarray(categories)
    total = (Dr**2).sum()

    def between(lab):
        e = 0.0
        for c in np.unique(lab):
            m = lab == c
            e += m.sum() * (Dr[m].mean(0) ** 2).sum()
        return e / total if total > 0 else np.nan

    obs = between(labels)
    rng = np.random.default_rng(seed)
    null = np.array([between(rng.permutation(labels)) for _ in range(n_perm)])
    return {"between_category_frac": float(obs), "null_mean": float(null.mean()),
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm))}


def summarise_null(observed: dict[str, float], null: list[dict[str, float]]) -> dict[str, dict]:
    out = {}
    for k, v in observed.items():
        xs = np.array([d.get(k, np.nan) for d in null], dtype=np.float64)
        xs = xs[np.isfinite(xs)]
        if not len(xs) or not np.isfinite(v):
            continue
        out[k] = {
            "observed": float(v), "null_mean": float(xs.mean()), "null_sd": float(xs.std()),
            "null_q95": float(np.quantile(xs, 0.95)),
            "p_greater": float((1 + (xs >= v).sum()) / (1 + len(xs))),
            "z": float((v - xs.mean()) / xs.std()) if xs.std() > 0 else float("nan"),
        }
    if "g_norm_sq" in out:
        # noise-corrected norm: ||g||^2 minus its expectation under the null
        c = out["g_norm_sq"]["observed"] - out["g_norm_sq"]["null_mean"]
        out["g_norm_debiased"] = {"observed": float(np.sqrt(max(c, 0.0)))}
    return out


def summarise_boot(boot: list[dict[str, float]], level: float = 0.95) -> dict[str, dict]:
    out = {}
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    for k in boot[0] if boot else []:
        xs = np.array([d.get(k, np.nan) for d in boot], dtype=np.float64)
        xs = xs[np.isfinite(xs)]
        if len(xs):
            out[k] = {"mean": float(xs.mean()), "lo": float(np.quantile(xs, lo)),
                      "hi": float(np.quantile(xs, hi)), "sd": float(xs.std())}
    return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def alignment_with(D: np.ndarray, g_ref: np.ndarray, roles: list[str]) -> dict:
    per = {r: cos(d, g_ref) for r, d in zip(roles, D)}
    return {"per_role_cos": per, "mean_role_cos": float(np.nanmean(list(per.values()))) if per else float("nan"),
            "cos_g": cos(D.mean(0), g_ref) if len(D) else float("nan"),
            "norm_ratio": float(np.linalg.norm(D.mean(0)) / np.linalg.norm(g_ref)) if len(D) and np.linalg.norm(g_ref) > 0 else float("nan")}


