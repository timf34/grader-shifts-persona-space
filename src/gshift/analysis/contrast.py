"""Linear contrasts over condition centroids, on matched units.

A contrast is a set of condition weights within ONE question group (e.g. evaluator main
effect = 1/2 (auto_correct + auto_quality) - 1/2 (human_correct + human_quality)). For
role r and unit u (question, prompt variant, sample) the per-unit contrast vector is

    E[r, u] = sum_c w_c h[r, c, u]          (defined iff every c with w_c != 0 is valid)

and the role displacement is Delta_r = mean over valid units of E[r, u] -- i.e. the
balanced difference of matched condition means, taken on RAW centroids (never after
per-condition centring or normalisation). A pooled contrast is an explicit, labelled
weighted average of group contrasts ("parts"): Delta_r = sum_p a_p mean_{u in p} E[r, u].

Null: paired sign flip of E[r, u] (for a two-condition contrast this is exactly the paired
condition-label swap). Bootstrap: roles with replacement, and questions with replacement
WITHIN each part, the same draw for every role and condition (shared-question and
repeated-prompt structure preserved: all prompt variants of a drawn question come along).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .cells import Cells
from .stats import gram_stats


@dataclass
class ContrastData:
    name: str
    label: str
    roles: list[str]
    units: list[tuple[str, int, int]]
    part: np.ndarray                       # [U] question group of each unit
    part_weight: dict[str, float]
    E: np.ndarray                          # [R, U, H] float32 (0 where invalid)
    M: np.ndarray                          # [R, U] bool
    qmeta: dict[str, dict] = field(default_factory=dict)
    categories: dict[str, str] = field(default_factory=dict)
    wording: np.ndarray | None = None      # [U] wording id of the unit's block (discovery grid)
    positions: dict | None = None          # {"A": [R, U, H], "B": ...} state positions (two-state contrasts)

    @property
    def H(self) -> int:
        return self.E.shape[-1]

    def ri(self, r: str) -> int:
        return self.roles.index(r)

    def unit_attr(self, key: str) -> np.ndarray:
        return np.array([self.qmeta.get(u[0], {}).get(key, "?") for u in self.units])

    def unit_question(self) -> np.ndarray:
        return np.array([u[0] for u in self.units])

    def subset_roles(self, roles: list[str]) -> "ContrastData":
        idx = [self.ri(r) for r in roles]
        pos = None if self.positions is None else {k: v[idx] for k, v in self.positions.items()}
        return ContrastData(self.name, self.label, list(roles), self.units, self.part, self.part_weight,
                            self.E[idx], self.M[idx], self.qmeta, self.categories, self.wording, pos)

    def subset_units(self, keep: np.ndarray) -> "ContrastData":
        keep = np.asarray(keep, bool)
        pos = None if self.positions is None else {k: v[:, keep] for k, v in self.positions.items()}
        parts = set(self.part[keep].tolist())
        pw = {p: w for p, w in self.part_weight.items() if p in parts}
        tot = sum(pw.values())
        pw = {p: w / tot for p, w in pw.items()} if tot > 0 else pw
        return ContrastData(self.name, self.label, self.roles, [u for u, k in zip(self.units, keep) if k],
                            self.part[keep], pw, self.E[:, keep], self.M[:, keep], self.qmeta, self.categories,
                            None if self.wording is None else self.wording[keep], pos)

    def n_pairs(self) -> np.ndarray:
        return self.M.sum(1)


def two_state(weights: dict[str, float]) -> tuple[dict[str, float], dict[str, float]] | None:
    """Split a contrast into state A (positive weights) and state B (negative), each summing
    to 1, if it is a difference of two weighted positions; None otherwise (interaction)."""
    pos = {c: w for c, w in weights.items() if w > 0}
    neg = {c: -w for c, w in weights.items() if w < 0}
    if not pos or not neg or not np.isclose(sum(pos.values()), 1) or not np.isclose(sum(neg.values()), 1):
        return None
    return pos, neg


def from_cells(cells: Cells, name: str, weights: dict[str, float], group: str, label: str | None = None,
               wording: np.ndarray | None = None, keep_positions: bool = False) -> ContrastData:
    """Contrast for one question group from its cell cube (cells.conds must contain the weights' keys)."""
    support = [c for c, w in weights.items() if w != 0]
    M = np.all(np.stack([cells.N[:, cells.ci(c)] > 0 for c in support]), 0)        # [R, U]
    E = np.zeros((len(cells.roles), len(cells.units), cells.H), dtype=np.float32)
    for c in support:
        E += np.float32(weights[c]) * cells.S[:, cells.ci(c)]
    E *= M[..., None]
    pos = None
    if keep_positions:
        ts = two_state(weights)
        if ts is not None:
            pos = {}
            for k, side in zip(("A", "B"), ts):
                P = np.zeros_like(E)
                for c, w in side.items():
                    P += np.float32(w) * cells.S[:, cells.ci(c)]
                pos[k] = P * M[..., None]
    part = np.array([group] * len(cells.units))
    return ContrastData(name, label or f"{group}: " + " ".join(f"{w:+g}*{c}" for c, w in weights.items()),
                        list(cells.roles), list(cells.units), part, {group: 1.0}, E, M, cells.qmeta,
                        cells.categories, wording, pos)


def pooled(name: str, label: str, parts: dict[str, ContrastData]) -> ContrastData:
    """Equal-weight pooled contrast over groups (units are disjoint across groups)."""
    first = next(iter(parts.values()))
    for cd in parts.values():
        if cd.roles != first.roles:
            raise ValueError("pooled parts must share roles")
    w = 1.0 / len(parts)
    E = np.concatenate([cd.E for cd in parts.values()], axis=1)
    M = np.concatenate([cd.M for cd in parts.values()], axis=1)
    part = np.concatenate([cd.part for cd in parts.values()])
    units = [u for cd in parts.values() for u in cd.units]
    wording = None
    if all(cd.wording is not None for cd in parts.values()):
        wording = np.concatenate([cd.wording for cd in parts.values()])
    pos = None
    if all(cd.positions is not None for cd in parts.values()):
        pos = {k: np.concatenate([cd.positions[k] for cd in parts.values()], axis=1) for k in ("A", "B")}
    return ContrastData(name, label, first.roles, units, part, {p: w for p in parts}, E, M,
                        first.qmeta, first.categories, wording, pos)


# ---------------------------------------------------------------------------
# displacements and resampling
# ---------------------------------------------------------------------------

def disp(cd: ContrastData, w: np.ndarray | None = None) -> np.ndarray:
    """Delta [R, H] (float64), optional unit weights w [U] (question bootstrap)."""
    w = np.ones(len(cd.units)) if w is None else np.asarray(w, np.float64)
    out = np.zeros((len(cd.roles), cd.H))
    for p, a in cd.part_weight.items():
        sel = cd.part == p
        wm = cd.M[:, sel] * w[sel][None]                      # [R, Up]
        num = np.einsum("ru,ruh->rh", wm.astype(np.float32), cd.E[:, sel]).astype(np.float64)
        den = wm.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out += a * np.nan_to_num(num / den[:, None])
    return out


def state_position(cd: ContrastData, state: str, w: np.ndarray | None = None) -> np.ndarray:
    """Matched-unit centroid of state A or B [R, H] (two-state contrasts only)."""
    if cd.positions is None:
        raise ValueError(f"{cd.name}: no state positions (not a two-state contrast)")
    tmp = ContrastData(cd.name, cd.label, cd.roles, cd.units, cd.part, cd.part_weight, cd.positions[state], cd.M)
    return disp(tmp, w)


def permutation_null(cd: ContrastData, n_perm: int, seed: int = 0, refs: dict[str, np.ndarray] | None = None,
                     batch: int = 100, stat_fn: Callable[[np.ndarray], dict] | None = None) -> list[dict]:
    """Paired sign-flip null: each valid (role, unit) contrast vector flips sign w.p. 1/2."""
    rng = np.random.default_rng(seed)
    refs = refs or {}
    vnorm = {k: float(np.linalg.norm(v)) for k, v in refs.items()}
    R = len(cd.roles)
    parts = [(cd.part == p, a) for p, a in cd.part_weight.items()]
    dens = [np.maximum(cd.M[:, sel].sum(1), 1)[:, None, None] for sel, _ in parts]
    out = []
    for start in range(0, n_perm, batch):
        P = min(batch, n_perm - start)
        D = np.zeros((R, P, cd.H))
        for (sel, a), den in zip(parts, dens):
            F = rng.choice(np.array([-1.0, 1.0], np.float32), size=(R, P, int(sel.sum())))
            D += a * (np.matmul(F, cd.E[:, sel]).astype(np.float64) / den)
        D = D.transpose(1, 0, 2)
        for p in range(P):
            Dp = D[p]
            out.append(stat_fn(Dp) if stat_fn else
                       gram_stats(Dp @ Dp.T, Dv={k: Dp @ v for k, v in refs.items()}, vnorm=vnorm))
    return out


def question_weights(rng, cd: ContrastData) -> np.ndarray:
    """Resample questions with replacement within each part; all units of a drawn question
    (its prompt variants) carry its multiplicity; same draw for all roles/conditions."""
    uq = cd.unit_question()
    w = np.zeros(len(uq))
    for p in cd.part_weight:
        sel = cd.part == p
        qs = np.unique(uq[sel])
        draw = rng.choice(qs, size=len(qs), replace=True)
        mult = dict(zip(*np.unique(draw, return_counts=True)))
        w[sel] = [mult.get(q, 0) for q in uq[sel]]
    return w


def bootstrap(cd: ContrastData, n_boot: int, seed: int = 0, refs: dict[str, np.ndarray] | None = None,
              stat_fn: Callable[[np.ndarray, np.ndarray], dict] | None = None) -> list[dict]:
    rng = np.random.default_rng(seed)
    refs = refs or {}
    vnorm = {k: float(np.linalg.norm(v)) for k, v in refs.items()}
    R = len(cd.roles)
    out = []
    for _ in range(n_boot):
        w = question_weights(rng, cd)
        ridx = rng.integers(0, R, size=R)
        D = disp(cd, w)[ridx]
        out.append(stat_fn(D, ridx) if stat_fn else
                   gram_stats(D @ D.T, origin=ridx, Dv={k: D @ v for k, v in refs.items()}, vnorm=vnorm))
    return out
