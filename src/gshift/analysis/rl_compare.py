"""Cross-checkpoint comparison along ONE RL trajectory from each checkpoint's small saved
analysis files (role displacements D, condition centroids, judge scores) -- never the
activation arrays.

Resampling unit: personas. The same discovery roles exist in every checkpoint, so ONE draw
of role indices is applied to every checkpoint (paired bootstrap); a CI on a difference
between checkpoints therefore accounts for the shared persona sample. Questions are held
fixed (every checkpoint's centroids average the same discovery questions), so these CIs
describe persona-sampling uncertainty only, not question sampling.

Every statistic below is a function of an R x R Gram (or pairwise-distance) matrix, so a
bootstrap replicate only re-indexes that matrix. Copies of the same role in a replicate
(same origin) are excluded from pairwise and leave-one-out terms, as in stats.gram_stats.

Raw coordinates are NOT assumed to align across checkpoints: landscape comparisons use
within-checkpoint geometry only (distance matrices, PC1 role scores, distances to the
default Assistant); raw cross-checkpoint cosines are reported as a sanity check.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

# ---------------------------------------------------------------------------
# resampling helpers
# ---------------------------------------------------------------------------


def boot_idx(n: int, n_boot: int, seed: int) -> np.ndarray:
    """[n_boot, n] role indices drawn with replacement (row 0 is NOT the identity)."""
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def ident(n: int) -> np.ndarray:
    return np.arange(n)[None]


def resample_sq(M: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """M [R, R], idx [B, R] -> [B, R, R] with rows and columns re-indexed."""
    return M[idx[:, :, None], idx[:, None, :]]


def distinct(idx: np.ndarray) -> np.ndarray:
    """[B, R, R] True where the two entries come from different original roles."""
    return idx[:, :, None] != idx[:, None, :]


def ci(x: np.ndarray, level: float = 0.95) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [float("nan"), float("nan")]
    a = (1 - level) / 2
    return [float(np.quantile(x, a)), float(np.quantile(x, 1 - a))]


def summarise(obs: float, boot: np.ndarray) -> dict:
    return {"est": float(obs), "ci": ci(boot), "boot_sd": float(np.nanstd(boot))}


def diff_summary(obs_a: float, obs_b: float, boot_a: np.ndarray, boot_b: np.ndarray) -> dict:
    """b - a with the paired-bootstrap CI (boot_a, boot_b from the SAME role draws)."""
    bs = np.asarray(boot_b, float) - np.asarray(boot_a, float)
    lo, hi = ci(bs)
    fin = bs[np.isfinite(bs)]
    p = float(min(1.0, 2 * min((fin <= 0).mean(), (fin >= 0).mean()))) if len(fin) else float("nan")
    return {"est": float(obs_b - obs_a), "ci": [lo, hi], "excludes_0": bool(lo > 0 or hi < 0), "p_boot": p}


def jack_idx(n: int) -> np.ndarray:
    """[n, n-1] leave-one-out index sets."""
    return np.array([np.delete(np.arange(n), i) for i in range(n)])


def jack_summary(obs: float, jv: np.ndarray) -> dict:
    """Leave-one-persona-out jackknife: SE and a normal 95% CI. Used for spectrum-based
    statistics (participation ratio, PC1 share, PC1 alignment), for which the percentile
    bootstrap is biased (duplicated personas lower the effective rank of a resample)."""
    jv = np.asarray(jv, float)
    n = len(jv)
    se = float(np.sqrt((n - 1) / n * np.sum((jv - jv.mean()) ** 2)))
    return {"est": float(obs), "ci": [float(obs - 1.96 * se), float(obs + 1.96 * se)], "se_jack": se, "method": "jackknife"}


def jack_diff(obs_a: float, obs_b: float, ja: np.ndarray, jb: np.ndarray) -> dict:
    """b - a with a paired jackknife CI (same persona left out in both)."""
    d = jack_summary(obs_b - obs_a, np.asarray(jb) - np.asarray(ja))
    d["excludes_0"] = bool(d["ci"][0] > 0 or d["ci"][1] < 0)
    return d


# ---------------------------------------------------------------------------
# shared-shift statistics (size, leave-one-out alignment)
# ---------------------------------------------------------------------------


def shift_stats(G: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """From G = D D^T [R, R] and role draws idx [B, R]: ||mean_r D_r|| and the mean
    leave-one-out cosine cos(D_r, mean of the roles with a different origin), per draw.
    For idx = identity this equals stats.gram_stats' g_norm and mean_loo_cos."""
    Gb = resample_sq(G, idx)
    R = idx.shape[1]
    g = np.sqrt(np.clip(Gb.sum((1, 2)), 0, None)) / R
    M = distinct(idx).astype(np.float64)
    k = M.sum(2)
    norms = np.sqrt(np.clip(np.einsum("bii->bi", Gb), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        dot = (Gb * M).sum(2) / k
        sq = np.einsum("bij,bjl,bil->bi", M, Gb, M) / k**2
        loo = dot / (norms * np.sqrt(np.clip(sq, 0, None)))
    return g, np.nanmean(loo, 1)


def pair_cross(XA: np.ndarray, XB: np.ndarray | None = None) -> np.ndarray:
    """S[i, j] = <xA_i - xA_j, xB_i - xB_j> (squared distance when XB is None). Inputs are
    centred first (pairwise differences are unchanged; better float64 precision)."""
    XA = XA - XA.mean(0)
    XB = XA if XB is None else XB - XB.mean(0)
    Gm = XA @ XB.T
    d = np.diag(Gm)
    return d[:, None] + d[None, :] - Gm - Gm.T


def mean_pair(S: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Mean of S over pairs of distinct-origin roles, per draw [B]."""
    Sb = resample_sq(S, idx)
    M = distinct(idx)
    return (Sb * M).sum((1, 2)) / M.sum((1, 2))


# ---------------------------------------------------------------------------
# persona landscape
# ---------------------------------------------------------------------------


def centred_gram(G: np.ndarray) -> np.ndarray:
    """Double-centre a batch of Gram matrices [..., n, n]."""
    rm = G.mean(-1, keepdims=True)
    return G - rm - np.swapaxes(rm, -1, -2) + G.mean((-1, -2), keepdims=True)


def landscape_from_gram(G: np.ndarray, gd: np.ndarray, dd: float, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Landscape statistics of role centroids X (rows) and default-Assistant centroid d,
    given G = X X^T [R, R], gd = X d [R], dd = d.d (X, d centred by the same vector).
    Per draw: total variance (mean squared distance to the role centroid), PCA eigenvalue
    spectrum, participation ratio, PC1 share, |cos(PC1, d - mean_role)| and the default's
    distance to the role centroid."""
    Gb = resample_sq(G, idx)                                     # [B, R, R]
    R = idx.shape[1]
    Gc = centred_gram(Gb)
    w, U = np.linalg.eigh(Gc)
    w, U = np.clip(w[:, ::-1], 0, None), U[:, :, ::-1]
    tot = w.sum(1)
    var = tot / R
    pr = tot**2 / (w**2).sum(1)
    pc1 = w[:, 0] / tot
    # a = d - mean_b ; Xc a for the resampled rows, all from Gram quantities
    gdb = gd[idx]                                                 # [B, R]  x_i . d
    xm = Gb.mean(2)                                               # [B, R]  x_i . mean
    md = gdb.mean(1)                                              # mean . d
    mm = Gb.mean((1, 2))                                          # mean . mean
    Xa = gdb - xm - md[:, None] + mm[:, None]                     # (x_i - mean) . (d - mean)
    a2 = dd - 2 * md + mm                                         # ||d - mean||^2
    u1 = U[:, :, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_pc1 = np.abs(np.einsum("bi,bi->b", u1, Xa)) / np.sqrt(w[:, 0] * a2)
        dist_def = np.sqrt(np.clip(a2, 0, None)) / np.sqrt(var)
    return {"var": var, "spread": np.sqrt(var), "evr": w / tot[:, None], "pr": pr, "pc1_share": pc1,
            "abs_cos_pc1_assistant": cos_pc1, "default_dist_over_spread": dist_def, "pc1_scores": u1 * np.sqrt(w[:, :1])}


def pca(X: np.ndarray) -> dict[str, np.ndarray]:
    """PCA of the rows of X via the Gram matrix: components [k, H], eigenvalue fractions,
    centre."""
    c = X.mean(0)
    Xc = X - c
    w, U = np.linalg.eigh(Xc @ Xc.T)
    w, U = w[::-1], U[:, ::-1]
    keep = w > w[0] * 1e-10
    comps = (U[:, keep].T @ Xc) / np.sqrt(w[keep])[:, None]
    return {"components": comps, "evr": w[keep] / w[keep].sum(), "center": c}


def upper(M: np.ndarray) -> np.ndarray:
    return M[np.triu_indices(len(M), 1)]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    return float(np.corrcoef(rx, ry)[0, 1])


def rsa_boot(DA: np.ndarray, DB: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Spearman correlation of two distance matrices over distinct-origin pairs i < j of
    each role draw."""
    out = np.empty(len(idx))
    iu = np.triu_indices(idx.shape[1], 1)
    for b, ix in enumerate(idx):
        i, j = ix[iu[0]], ix[iu[1]]
        keep = i != j
        out[b] = spearman(DA[i[keep], j[keep]], DB[i[keep], j[keep]])
    return out


def dist(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Euclidean distances between rows of X (and Y)."""
    Y = X if Y is None else Y
    c = X.mean(0)
    X, Y = X - c, Y - c
    d2 = (X**2).sum(1)[:, None] + (Y**2).sum(1)[None] - 2 * X @ Y.T
    return np.sqrt(np.clip(d2, 0, None))


def pc1_scores_boot(GA: np.ndarray, GB: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """|corr| of PC1 role scores fitted separately in two checkpoints on the same role draw."""
    out = np.empty(len(idx))
    for b, ix in enumerate(idx):
        sa = _pc1(GA[np.ix_(ix, ix)])
        sb = _pc1(GB[np.ix_(ix, ix)])
        out[b] = abs(np.corrcoef(sa, sb)[0, 1])
    return out


def _pc1(G: np.ndarray) -> np.ndarray:
    w, U = np.linalg.eigh(centred_gram(G))
    return U[:, -1] * np.sqrt(max(w[-1], 0))


# ---------------------------------------------------------------------------
# adherence (judge scores)
# ---------------------------------------------------------------------------


def unit_boot_mean(v: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Bootstrap means of v resampling its entries (units) with replacement."""
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for s in range(0, n_boot, 500):
        k = min(500, n_boot - s)
        out[s:s + k] = v[rng.integers(0, len(v), size=(k, len(v)))].mean(1)
    return out


def twoway_boot_mean(V: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """V [roles, questions] with NaN for missing cells: bootstrap of the cell mean drawing
    roles and questions independently with replacement (crossed design; conservative)."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(V)
    V0 = np.where(ok, V, 0.0)
    R, Q = V.shape
    out = np.empty(n_boot)
    for b in range(n_boot):
        wr = np.bincount(rng.integers(0, R, R), minlength=R).astype(float)
        wq = np.bincount(rng.integers(0, Q, Q), minlength=Q).astype(float)
        num = wr @ V0 @ wq
        den = wr @ ok @ wq
        out[b] = num / den if den > 0 else np.nan
    return out
