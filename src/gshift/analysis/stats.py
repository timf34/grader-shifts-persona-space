"""Is the shift shared? Statistics of the role-displacement matrix D [R, H], its null and CIs.

Nearly every statistic is a function of the Gram matrix G = D D^T (R x R) and of D v for
a few reference vectors v, so the permutation null and the bootstrap only ever need
G for each resample.

Null: paired condition-label permutation. Under H0 (the statement has no effect) the two
conditions of a unit are exchangeable, so each (role, unit) independently swaps its
labels with probability 1/2. Units with a row in only one condition swap that row's
label, which is also exchangeable under H0.

Bootstrap: roles and questions resampled with replacement, the SAME question multiset
for every role and condition (preserving the shared-question design). Pairs of copies of
the same role are excluded from pairwise / leave-one-out statistics.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .cells import Cells, centroids


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


# ---------------------------------------------------------------------------
# resampling (all functions take a MATCHED two-condition view: cells.conds == [a, b])
# ---------------------------------------------------------------------------

def _check_pair(cells: Cells) -> None:
    if len(cells.conds) != 2:
        raise ValueError("expected a matched two-condition view (Cells.matched(a, b))")


def permutation_null(cells: Cells, n_perm: int, seed: int = 0, refs: dict[str, np.ndarray] | None = None,
                     batch: int = 100, stat_fn: Callable[[np.ndarray], dict] | None = None) -> list[dict[str, float]]:
    """Paired label-swap null for a - b: each (role, unit) swaps its two rows w.p. 1/2."""
    _check_pair(cells)
    Sa, Sb = cells.S[:, 0], cells.S[:, 1]           # [R, U, H]
    Na, Nb = cells.N[:, 0], cells.N[:, 1]
    R, U, H = Sa.shape
    dS = (Sb - Sa).astype(np.float32)
    dN = Nb - Na
    Sa_tot, Sb_tot = Sa.sum(1, dtype=np.float64), Sb.sum(1, dtype=np.float64)
    Na_tot, Nb_tot = Na.sum(1), Nb.sum(1)
    rng = np.random.default_rng(seed)
    refs = refs or {}
    vnorm = {k: float(np.linalg.norm(v)) for k, v in refs.items()}
    out = []
    for start in range(0, n_perm, batch):
        P = min(batch, n_perm - start)
        F = rng.integers(0, 2, size=(R, P, U)).astype(np.float32)
        FdS = np.matmul(F, dS).astype(np.float64)    # [R, P, H] batched BLAS
        FdN = np.einsum("rpu,ru->rp", F.astype(np.float64), dN)
        with np.errstate(invalid="ignore", divide="ignore"):
            D = ((Sa_tot[:, None] + FdS) / (Na_tot[:, None] + FdN)[..., None]
                 - (Sb_tot[:, None] - FdS) / (Nb_tot[:, None] - FdN)[..., None]).transpose(1, 0, 2)
        D = np.nan_to_num(D)
        for p in range(P):
            Dp = D[p]
            if stat_fn is not None:
                out.append(stat_fn(Dp))
            else:
                out.append(gram_stats(Dp @ Dp.T, Dv={k: Dp @ v for k, v in refs.items()}, vnorm=vnorm))
    return out


def question_weights(rng, uq: np.ndarray) -> np.ndarray:
    """Resample questions with replacement; every unit of a question (all its prompt
    variants) carries that question's multiplicity -- preserves the repeated-prompt and
    shared-question structure (same draw for all roles and conditions)."""
    qs = np.unique(uq)
    draw = rng.choice(qs, size=len(qs), replace=True)
    mult = dict(zip(*np.unique(draw, return_counts=True)))
    return np.array([mult.get(q, 0) for q in uq], dtype=np.float64)


def bootstrap(cells: Cells, n_boot: int, seed: int = 0, refs: dict[str, np.ndarray] | None = None,
              resample_roles: bool = True, resample_questions: bool = True) -> list[dict[str, float]]:
    """Two-way (roles x questions) bootstrap of gram_stats for the matched view a - b."""
    _check_pair(cells)
    rng = np.random.default_rng(seed)
    uq = cells.unit_question()
    R = len(cells.roles)
    refs = refs or {}
    vnorm = {k: float(np.linalg.norm(v)) for k, v in refs.items()}
    out = []
    for _ in range(n_boot):
        w = question_weights(rng, uq) if resample_questions else np.ones(len(uq))
        ridx = rng.integers(0, R, size=R) if resample_roles else np.arange(R)
        mu = centroids(cells, w)                    # all roles once, then pick copies
        D = np.nan_to_num(mu[ridx, 0] - mu[ridx, 1])
        out.append(gram_stats(D @ D.T, origin=ridx, Dv={k: D @ v for k, v in refs.items()}, vnorm=vnorm))
    return out


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
# generalisation and stability
# ---------------------------------------------------------------------------

def cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def disp(cells: Cells, w: np.ndarray | None = None) -> np.ndarray:
    """Delta [R, H] for a matched two-condition view."""
    _check_pair(cells)
    mu = centroids(cells, w)
    return np.nan_to_num(mu[:, 0] - mu[:, 1])


def alignment_with(D: np.ndarray, g_ref: np.ndarray, roles: list[str]) -> dict:
    per = {r: cos(d, g_ref) for r, d in zip(roles, D)}
    return {"per_role_cos": per, "mean_role_cos": float(np.nanmean(list(per.values()))) if per else float("nan"),
            "cos_g": cos(D.mean(0), g_ref) if len(D) else float("nan"),
            "norm_ratio": float(np.linalg.norm(D.mean(0)) / np.linalg.norm(g_ref)) if len(D) and np.linalg.norm(g_ref) > 0 else float("nan")}


def alignment_null(cells: Cells, g_ref: np.ndarray, n_perm: int, seed: int = 0) -> dict:
    """Null for mean per-role cos(Delta_r, g_ref) on cells independent of g_ref."""
    obs = float(np.nanmean([cos(d, g_ref) for d in disp(cells)]))
    null = permutation_null(cells, n_perm, seed=seed,
                            stat_fn=lambda D: {"m": float(np.nanmean([cos(d, g_ref) for d in D]))})
    xs = np.array([d["m"] for d in null])
    return {"observed": obs, "null_mean": float(xs.mean()), "null_q95": float(np.quantile(xs, 0.95)),
            "p_greater": float((1 + (xs >= obs).sum()) / (1 + len(xs)))}


def split_half_stability(cells: Cells, n_splits: int, seed: int = 0, balance_key: str = "group") -> dict:
    """Shared direction from two balanced halves of the questions (balanced on `balance_key`,
    e.g. 10 original + 10 checkable each), roles and prompt variants fixed; cos(g1, g2)."""
    rng = np.random.default_rng(seed)
    uq = cells.unit_question()
    qs = sorted(set(uq))
    strata: dict[str, list[str]] = {}
    for q in qs:
        strata.setdefault(str(cells.qmeta.get(q, {}).get(balance_key, "?")), []).append(q)
    vals = []
    for _ in range(n_splits):
        half = set()
        for members in strata.values():
            perm = rng.permutation(len(members))
            half |= {members[i] for i in perm[: len(members) // 2]}
        in_half = np.array([q in half for q in uq])
        g1 = disp(cells.subset_units(in_half)).mean(0)
        g2 = disp(cells.subset_units(~in_half)).mean(0)
        vals.append(cos(g1, g2))
    v = np.array(vals)
    return {"mean_cos": float(np.nanmean(v)), "min_cos": float(np.nanmin(v)), "sd": float(np.nanstd(v)),
            "n_splits": n_splits, "strata_sizes": {k: len(m) for k, m in strata.items()}}


def family_leave_one_out(cells: Cells) -> dict:
    """g from the other question families vs displacements on the held-out family."""
    fam = cells.unit_attr("family")
    out = {}
    for f in sorted(set(fam)):
        inside = fam == f
        D_in, D_out = disp(cells.subset_units(inside)), disp(cells.subset_units(~inside))
        g_out = D_out.mean(0)
        out[f] = {"mean_role_cos_with_g_other": float(np.nanmean([cos(d, g_out) for d in D_in])),
                  "cos_g_family_g_other": cos(D_in.mean(0), g_out)}
    return {"families": out, "mean_cos_g_family_g_other": float(np.nanmean([v["cos_g_family_g_other"]
                                                                              for v in out.values()]))}


def crossfit_standardised_effect(cells: Cells, base_index: int = 1) -> dict:
    """Leave-one-role-out standardised shift: (Delta_r . g_hat_{-r}) / sd_r, where sd_r is the
    SD of role r's baseline responses along g_hat_{-r}. Cross-fitting keeps the direction
    independent of the role it is evaluated on; the unit (within-role baseline SD) makes the
    number comparable across models."""
    D = disp(cells)
    effs, projs = [], []
    for r in range(len(cells.roles)):
        g = np.delete(D, r, 0).mean(0)
        gn = np.linalg.norm(g)
        if gn == 0:
            continue
        gh = g / gn
        n = cells.N[r, base_index]
        rows = cells.S[r, base_index][n > 0].astype(np.float64)
        if len(rows) < 3:
            continue
        sd = float((rows @ gh).std(ddof=1))
        e = float(D[r] @ gh)
        effs.append(e / sd if sd > 0 else np.nan)
        projs.append(e)
    return {"mean_d": float(np.nanmean(effs)) if effs else float("nan"), "per_role_d": effs,
            "mean_proj": float(np.nanmean(projs)) if projs else float("nan")}
