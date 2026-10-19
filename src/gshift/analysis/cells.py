"""Cell representation of one (representation, layer) slice.

S[r, c, u, :] = activation of the valid row for role r, condition c, unit u (else 0)
N[r, c, u]    = 1 if that row exists and is valid (0/1 with one sample per prompt)
with unit u = (question id, prompt variant, sample). Every condition of a role shares the
same units, so with unit weights w_u (question bootstrap)

    mu[r, c] = sum_u w_u S[r, c, u] / sum_u w_u N[r, c, u]   ( = mean_{q,p} h_{r,c,q,p} )

Contrasts use MATCHED units only: a unit enters a - b iff it is valid in both a and b.
Differences are taken on these raw centroids -- never after per-condition centring or
normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Cells:
    roles: list[str]
    conds: list[str]
    units: list[tuple[str, int, int]]      # (question_id, prompt_variant, sample)
    S: np.ndarray                          # [R, C, U, H] float32
    N: np.ndarray                          # [R, C, U]    float64 (0/1)
    categories: dict[str, str] = field(default_factory=dict)
    qmeta: dict[str, dict] = field(default_factory=dict)   # question id -> {group, family, split, ...}

    @property
    def H(self) -> int:
        return self.S.shape[-1]

    def ci(self, cond: str) -> int:
        return self.conds.index(cond)

    def ri(self, role: str) -> int:
        return self.roles.index(role)

    def unit_question(self) -> np.ndarray:
        return np.array([u[0] for u in self.units])

    def unit_attr(self, key: str) -> np.ndarray:
        return np.array([self.qmeta.get(u[0], {}).get(key, "?") for u in self.units])

    def subset_roles(self, roles: list[str]) -> "Cells":
        idx = [self.ri(r) for r in roles]
        return Cells(list(roles), self.conds, self.units, self.S[idx], self.N[idx], self.categories, self.qmeta)

    def subset_units(self, keep: np.ndarray) -> "Cells":
        keep = np.asarray(keep, bool)
        return Cells(self.roles, self.conds, [u for u, k in zip(self.units, keep) if k],
                     self.S[:, :, keep], self.N[:, :, keep], self.categories, self.qmeta)

    def matched(self, a: str, b: str, extra_mask: np.ndarray | None = None) -> "Cells":
        """Two-condition view [a, b] keeping only units valid in BOTH (per role).
        extra_mask [R, U] can further require e.g. role adherence in both."""
        ia, ib = self.ci(a), self.ci(b)
        ok = (self.N[:, ia] > 0) & (self.N[:, ib] > 0)
        if extra_mask is not None:
            ok &= extra_mask
        m = ok.astype(np.float64)
        S = self.S[:, [ia, ib]] * m[:, None, :, None].astype(self.S.dtype)
        N = self.N[:, [ia, ib]] * m[:, None, :]
        return Cells(self.roles, [a, b], self.units, S, N, self.categories, self.qmeta)

    def n_pairs(self) -> np.ndarray:
        """Matched pairs per role (for a 2-condition matched view)."""
        return np.minimum(self.N[:, 0], self.N[:, 1]).sum(1)


def build_cells(meta: pd.DataFrame, X: np.ndarray, valid: np.ndarray, roles: list[str], conds: list[str],
                categories: dict[str, str] | None = None, qmeta: dict[str, dict] | None = None,
                units: list[tuple[str, int, int]] | None = None) -> Cells:
    """meta: DataFrame aligned with X [N, H]; valid: bool [N]."""
    if units is None:
        units = sorted({(str(q), int(p), int(s)) for q, p, s in
                        zip(meta["question_id"], meta["prompt_variant"], meta["sample"])})
    uidx = {u: i for i, u in enumerate(units)}
    ridx = {r: i for i, r in enumerate(roles)}
    cidx = {c: i for i, c in enumerate(conds)}
    R, C, U, H = len(roles), len(conds), len(units), X.shape[1]
    S = np.zeros((R, C, U, H), dtype=np.float32)
    N = np.zeros((R, C, U), dtype=np.float64)
    for i, (r, c, q, p, s, ok) in enumerate(zip(meta["role"], meta["condition"], meta["question_id"],
                                                meta["prompt_variant"], meta["sample"], valid)):
        if not ok or r not in ridx or c not in cidx:
            continue
        u = uidx.get((str(q), int(p), int(s)))
        if u is None:
            continue
        S[ridx[r], cidx[c], u] += X[i]
        N[ridx[r], cidx[c], u] += 1
    return Cells(list(roles), list(conds), list(units), S, N, dict(categories or {}), dict(qmeta or {}))


def centroids(cells: Cells, w: np.ndarray | None = None) -> np.ndarray:
    """mu [R, C, H] (float64) with optional unit weights w [U]. NaN where a cell is empty."""
    w = np.ones(len(cells.units)) if w is None else np.asarray(w, dtype=np.float64)
    num = np.tensordot(cells.S, w.astype(cells.S.dtype), axes=([2], [0])).astype(np.float64)
    den = cells.N @ w
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den[..., None]


def displacements(cells: Cells, a: str, b: str, w: np.ndarray | None = None, matched: bool = True) -> np.ndarray:
    """Delta [R, H] = mu[r, a] - mu[r, b] on matched units (raw, un-normalised)."""
    c = cells.matched(a, b) if matched else cells
    mu = centroids(c, w)
    return mu[:, c.ci(a)] - mu[:, c.ci(b)]


def unit_normalised_direction(D: np.ndarray, rel_eps: float = 1e-3) -> tuple[np.ndarray, list[int]]:
    """mean_r Delta_r/||Delta_r||, EXCLUDING near-zero displacements (norm < rel_eps x median),
    whose direction is undefined. Returns (direction, excluded role indices)."""
    n = np.linalg.norm(D, axis=1)
    med = np.median(n) if len(n) else 0.0
    keep = n > rel_eps * med if med > 0 else n > 0
    if not keep.any():
        return np.zeros(D.shape[1]), list(range(len(D)))
    return (D[keep] / n[keep, None]).mean(0), [int(i) for i in np.where(~keep)[0]]
