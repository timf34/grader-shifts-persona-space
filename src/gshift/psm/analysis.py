"""PSM analysis (local, CPU): Track A persona decision profiles, convergence and landscape
prediction; Track B steering effects by context; projections before the decision.

All CIs bootstrap ITEMS (the scenario instances), with the same resamples for every cell of a
comparison, so differences are paired. Persona-level statistics treat personas as fixed (the
persona set is frozen); permutation nulls shuffle persona labels within items.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

N_BOOT = 2000


def _ci(x: np.ndarray) -> list[float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return [float(v) for v in np.percentile(x, [2.5, 97.5])] if len(x) else [float("nan")] * 2


def _f(x) -> float | None:
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------------------
# matrices: persona x item rates per context
# ---------------------------------------------------------------------------

def rate_matrix(rows: list[dict], personas: list[str], items: list[str], value: str = "misreport",
                valid_key: str = "valid") -> np.ndarray:
    """[P, I] mean of `value` over valid samples (NaN where none)."""
    pi = {p: i for i, p in enumerate(personas)}
    ii = {it: j for j, it in enumerate(items)}
    s = np.zeros((len(personas), len(items)))
    n = np.zeros_like(s)
    for r in rows:
        if r.get(valid_key) and r.get(value) is not None and r["persona"] in pi and r["item_id"] in ii:
            s[pi[r["persona"]], ii[r["item_id"]]] += r[value]
            n[pi[r["persona"]], ii[r["item_id"]]] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def _nanmean(a, axis):
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, axis=axis)


def persona_rates(M: np.ndarray, boot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[P] point rates and [P, B] bootstrap rates (items resampled)."""
    return _nanmean(M, 1), _nanmean(M[:, boot], 2)


def spread(p: np.ndarray, d_idx: int) -> tuple[np.ndarray, np.ndarray]:
    """(SD across non-default personas, mean |p - p_default|), along axis 0."""
    others = [i for i in range(p.shape[0]) if i != d_idx]
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanstd(p[others], axis=0, ddof=1), np.nanmean(np.abs(p[others] - p[d_idx]), axis=0)


def perm_null_sd(M: np.ndarray, d_idx: int, n_perm: int = 1000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm)
    for k in range(n_perm):
        Mp = np.take_along_axis(M, np.argsort(rng.random(M.shape), axis=0), axis=0)
        out[k] = spread(_nanmean(Mp, 1), d_idx)[0]
    return out


def track_a(rows: list[dict], personas: list[str], contexts: list[str], kinds=("conflict",),
            value: str = "misreport", seed: int = 0) -> dict[str, Any]:
    """Persona profiles, separation and convergence on the reporting assay (or questionnaire with
    value='choose_favoured')."""
    rows = [r for r in rows if r.get("kind") in kinds]
    items = sorted({r["item_id"] for r in rows})
    boot = np.random.default_rng(seed).integers(0, len(items), (N_BOOT, len(items)))
    d_idx = personas.index("default")
    by_ctx = {c: [r for r in rows if r["context"] == c] for c in contexts}
    mats = {c: rate_matrix(by_ctx[c], personas, items, value) for c in contexts if by_ctx[c]}
    out: dict[str, Any] = {"n_items": len(items), "personas": personas, "contexts": {}}
    ref = mats.get("base")
    ref_p, ref_b = persona_rates(ref, boot) if ref is not None else (None, None)
    def xsplit_sd(c, bidx=None):
        """Noise-free between-persona SD: covariance across (non-default) personas of the rates
        from sample 0 and sample 1 (independent draws), sqrt(max(0, .))."""
        m0 = rate_matrix([r for r in by_ctx[c] if r.get("sample") == 0], personas, items, value)
        m1 = rate_matrix([r for r in by_ctx[c] if r.get("sample") == 1], personas, items, value)
        if bidx is None:
            a, b = _nanmean(m0, 1)[:, None], _nanmean(m1, 1)[:, None]
        else:
            a, b = _nanmean(m0[:, bidx], 2), _nanmean(m1[:, bidx], 2)
        o = [i for i in range(len(personas)) if i != d_idx]
        a, b = a[o], b[o]
        cov = ((a - np.nanmean(a, 0)) * (b - np.nanmean(b, 0))).sum(0) / (len(o) - 1)
        return np.sqrt(np.maximum(cov, 0)), cov

    has_two = any(r.get("sample") == 1 for r in rows)
    xs = {}
    if has_two:
        for c in mats:
            xs[c] = (xsplit_sd(c), xsplit_sd(c, boot))
    for c, M in mats.items():
        p, pb = persona_rates(M, boot)
        sd, mad = spread(p, d_idx)
        sdb, madb = spread(pb, d_idx)
        null = perm_null_sd(M, d_idx, 500, seed)
        e = {"rates": {per: {"p": _f(p[i]), "ci95": _ci(pb[i])} for i, per in enumerate(personas)},
             "sd": _f(sd), "sd_ci95": _ci(sdb), "sd_null_mean": _f(null.mean()),
             "sd_perm_p": _f((1 + (null >= sd).sum()) / (1 + len(null))),
             "mad_default": _f(mad), "mad_ci95": _ci(madb),
             "mean_rate": _f(np.nanmean(p)), "mean_rate_ci95": _ci(_nanmean(pb, 0))}
        if c in xs:
            (s_pt, _), (s_b, _) = xs[c]
            e["sd_xsplit"] = _f(s_pt[0])
            e["sd_xsplit_ci95"] = _ci(s_b)
            if "base" in xs and c != "base":
                (r_pt, _), (r_b, _) = xs["base"]
                with np.errstate(divide="ignore", invalid="ignore"):
                    e["convergence_xsplit_ratio"] = _f(s_pt[0] / r_pt[0])
                    e["convergence_xsplit_ratio_ci95"] = _ci(s_b / r_b)
        if ref is not None and c != "base":
            rsd, rmad = spread(ref_p, d_idx)
            rsdb, rmadb = spread(ref_b, d_idx)
            with np.errstate(divide="ignore", invalid="ignore"):
                e["convergence_sd_ratio"] = _f(sd / rsd)
                e["convergence_sd_ratio_ci95"] = _ci(sdb / rsdb)
                e["convergence_mad_ratio"] = _f(mad / rmad)
                e["convergence_mad_ratio_ci95"] = _ci(madb / rmadb)
                e["rate_diff_vs_base"] = {per: {"d": _f(p[i] - ref_p[i]), "ci95": _ci(pb[i] - ref_b[i])}
                                          for i, per in enumerate(personas)}
                # toward the default under the SAME context: |p - p_def| shrinks per persona?
                dist, dist_ref = np.abs(p - p[d_idx]), np.abs(ref_p - ref_p[d_idx])
                distb, dist_refb = np.abs(pb - pb[d_idx]), np.abs(ref_b - ref_b[d_idx])
                e["per_persona_dist_to_default"] = {
                    per: {"dist": _f(dist[i]), "dist_base": _f(dist_ref[i]),
                          "diff_ci95": _ci(distb[i] - dist_refb[i])}
                    for i, per in enumerate(personas) if i != d_idx}
        out["contexts"][c] = e
    out["_mats"] = mats
    out["_items"] = items
    return out


def outcome_table(rows: list[dict], personas: list[str], contexts: list[str]) -> dict[str, Any]:
    """Validity, outcome mix, competence (control accuracy) and lengths per persona x context."""
    out: dict[str, Any] = {}
    for c in contexts:
        for p in personas:
            rs = [r for r in rows if r["context"] == c and r["persona"] == p]
            if not rs:
                continue
            conf = [r for r in rs if r["kind"] == "conflict"]
            ctrl = [r for r in rs if r["kind"] != "conflict"]
            oc = defaultdict(int)
            for r in conf:
                oc[r["outcome"]] += 1
            out.setdefault(c, {})[p] = {
                "n": len(rs), "valid": _f(np.mean([r["valid"] for r in rs])),
                "conflict_outcomes": {k: v / max(1, len(conf)) for k, v in sorted(oc.items())},
                "control_acc": _f(np.mean([r["outcome"] == "accurate" for r in ctrl])) if ctrl else None,
                "truncated": _f(np.mean([r["truncated"] for r in rs])),
                "answer_tokens_median": _f(np.median([r["n_answer_tokens"] or 0 for r in rs])),
                "reasoning_tokens_median": _f(np.median([r["n_reasoning_tokens"] or 0 for r in rs]))}
    return out


def by_pressure(rows: list[dict], contexts: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in contexts:
        for pr in ("mild", "strong", "explicit", "suggest"):
            rs = [r for r in rows if r["context"] == c and r["kind"] == "conflict" and r["pressure"] == pr and r["valid"]]
            if rs:
                out.setdefault(c, {})[pr] = {"misreport": _f(np.mean([r["misreport"] for r in rs])), "n": len(rs),
                                             "by_persona": {p: _f(np.mean([r["misreport"] for r in rs if r["persona"] == p]))
                                                            for p in sorted({r["persona"] for r in rs})}}
    return out


# ---------------------------------------------------------------------------
# landscape: activations on the assay prompts
# ---------------------------------------------------------------------------

def load_acts(npz: Path, layer: int) -> tuple[np.ndarray, list[dict]]:
    z = np.load(npz)
    li = list(z["layers"]).index(layer)
    meta = [json.loads(m) for m in z["meta"]]
    return z["acts"][:, li].astype(np.float32), meta


def _is(m: dict, which: str | None) -> bool:
    if which is None:
        return True
    return (m.get("task") == "rpt") if which == "rpt" else ("dimension" in m)


def persona_centroids(A: np.ndarray, meta: list[dict], personas: list[str], context: str,
                      which: str | None = None) -> np.ndarray:
    out = np.zeros((len(personas), A.shape[1]), np.float64)
    for i, p in enumerate(personas):
        idx = [k for k, m in enumerate(meta) if m["persona"] == p and m["context"] == context and _is(m, which)]
        out[i] = A[idx].astype(np.float64).mean(0)
    return out


def act_spread(C: np.ndarray, d_idx: int) -> tuple[float, float]:
    others = [i for i in range(len(C)) if i != d_idx]
    X = C[others]
    pair = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    return float(pair[np.triu_indices(len(X), 1)].mean()), float(np.linalg.norm(X - C[d_idx], axis=1).mean())


def mantel(D1: np.ndarray, D2: np.ndarray, n_perm: int = 5000, seed: int = 0) -> dict[str, float]:
    """Spearman correlation of upper triangles; permutation of rows/cols of D2 (persona labels)."""
    from scipy.stats import spearmanr

    iu = np.triu_indices(len(D1), 1)
    r = spearmanr(D1[iu], D2[iu]).correlation
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        q = rng.permutation(len(D2))
        null[k] = spearmanr(D1[iu], D2[q][:, q][iu]).correlation
    return {"rho": float(r), "perm_p": float((1 + (null >= r).sum()) / (1 + n_perm)), "null_mean": float(null.mean())}


def landscape(acts_npz: Path, layer: int, personas: list[str], contexts: list[str],
              mats: dict[str, np.ndarray], which: str | None = "rpt") -> dict[str, Any]:
    """Activation spread of persona centroids by context (convergence in representation) and
    whether activation distances predict behavioural distances (Mantel)."""
    A, meta = load_acts(acts_npz, layer)
    d_idx = personas.index("default")
    norm = float(np.linalg.norm(A.astype(np.float64), axis=1).mean())
    out: dict[str, Any] = {"layer": layer, "act_norm": norm, "contexts": {}}
    contexts = [c for c in contexts if any(m["context"] == c and _is(m, which) for m in meta)]
    C = {c: persona_centroids(A, meta, personas, c, which) for c in contexts}
    base_pair, base_def = act_spread(C["base"], d_idx)
    for c in contexts:
        pair, dd = act_spread(C[c], d_idx)
        e = {"pairwise_dist_rel": pair / norm, "dist_to_default_rel": dd / norm,
             "pairwise_ratio_vs_base": pair / base_pair, "dist_to_default_ratio_vs_base": dd / base_def}
        if c in mats:
            beh = mats[c] if mats[c].ndim == 2 and mats[c].shape[1] <= 12 else _nanmean(mats[c], 1)
            beh = beh.reshape(len(personas), -1)
            Db = np.sqrt(np.nansum((beh[:, None] - beh[None]) ** 2, -1))
            Da = np.sqrt(((C[c][:, None] - C[c][None]) ** 2).sum(-1))
            e["mantel_act_vs_behaviour"] = mantel(Da, Db)
            # does the base-context landscape predict the graded behaviour?
            Da0 = np.sqrt(((C["base"][:, None] - C["base"][None]) ** 2).sum(-1))
            e["mantel_baseacts_vs_behaviour"] = mantel(Da0, Db)
        out["contexts"][c] = e
    out["_centroids"] = C
    return out


# ---------------------------------------------------------------------------
# Track B
# ---------------------------------------------------------------------------

def item_rate_map(rows: list[dict]) -> dict[str, float]:
    acc: dict[str, list] = defaultdict(list)
    for r in rows:
        if r["kind"] == "conflict" and r["valid"]:
            acc[r["item_id"]].append(r["misreport"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


def track_b(rows_by: dict[tuple[str, str], list[dict]], choice: dict, contexts: list[str], n_random: int,
            seed: int = 0) -> dict[str, Any]:
    """rows_by[(context, condition)] -> scored rows. Effects vs none (same context) and the
    context x intervention interaction on the trait range r = rate(+s) - rate(-s)."""
    L, s = int(choice["layer"]), float(choice["strength"])
    fmt = lambda d, x: f"{d}@L{L}_{'+' if x >= 0 else '-'}{abs(x):g}"  # noqa: E731
    items = sorted({r["item_id"] for rs in rows_by.values() for r in rs if r["kind"] == "conflict"})
    boot = np.random.default_rng(seed).integers(0, len(items), (N_BOOT, len(items)))

    def vec(ctx, cond):
        m = item_rate_map(rows_by.get((ctx, cond), []))
        return np.array([m.get(i, np.nan) for i in items])

    def logit(p):
        p = np.clip(p, 0.02, 0.98)
        return np.log(p / (1 - p))

    out: dict[str, Any] = {"layer": L, "strength": s, "contexts": {}}
    ranges = {}
    for ctx in contexts:
        none = vec(ctx, "none")
        e: dict[str, Any] = {"none": {"rate": _f(np.nanmean(none)), "ci95": _ci(_nanmean(none[boot], 1))}}
        for d in ["trait"] + [f"random{k}" for k in range(n_random)] + ["traitdpo"]:
            for x in ([s / 2, s, -s / 2, -s] if d == "trait" else [s, -s]):
                v = vec(ctx, fmt(d, x))
                if np.all(np.isnan(v)):
                    continue
                diff = v - none
                e[fmt(d, x)] = {"rate": _f(np.nanmean(v)), "diff_vs_none": _f(np.nanmean(diff)),
                                "diff_ci95": _ci(_nanmean(diff[boot], 1))}
            p, m = vec(ctx, fmt(d, s)), vec(ctx, fmt(d, -s))
            if not (np.all(np.isnan(p)) or np.all(np.isnan(m))):
                r = p - m
                e[f"range_{d}"] = {"range": _f(np.nanmean(r)), "ci95": _ci(_nanmean(r[boot], 1)),
                                   "logit_range": _f(np.nanmean(logit(p) - logit(m)))}
                ranges[(ctx, d)] = r
        band = [abs(e[f"range_random{k}"]["range"]) for k in range(n_random) if f"range_random{k}" in e]
        e["random_band"] = _f(max(band)) if band else None
        # competence-filtered one-sided comparison (reported beside the pre-registered range): each
        # condition's validity and control accuracy; the band = max |rate - none| over random
        # conditions that keep valid >= 0.90 and control accuracy >= 0.85
        q = {}
        for cond in sorted({c for (cx, c) in rows_by if cx == ctx}):
            rs = rows_by[(ctx, cond)]
            conf = [r for r in rs if r["kind"] == "conflict" and r["valid"]]
            ctrl = [r for r in rs if r["kind"] != "conflict"]
            q[cond] = {"rate": _f(np.mean([r["misreport"] for r in conf])) if conf else None,
                       "valid": _f(np.mean([r["valid"] for r in rs])) if rs else None,
                       "control_acc": _f(np.mean([r["outcome"] == "accurate" for r in ctrl])) if ctrl else None,
                       "by_pressure": {pr: _f(np.mean([r["misreport"] for r in conf if r["pressure"] == pr]))
                                       for pr in sorted({r["pressure"] for r in conf})}}
            q[cond]["competent"] = bool((q[cond]["valid"] or 0) >= 0.90 and (q[cond]["control_acc"] or 0) >= 0.85)
        n0 = q.get("none", {}).get("rate") or 0.0
        cb = [abs(v["rate"] - n0) for c, v in q.items() if c.startswith("random") and v["competent"] and v["rate"] is not None
              and abs(abs(float(c.rsplit("_", 1)[1])) - s) < 1e-9 and c.split("@L")[1].startswith(str(L) + "_")]
        e["conditions"] = q
        e["competent_random_band"] = _f(max(cb)) if cb else None
        out["contexts"][ctx] = e
    # interaction: does grading change the trait's effect? (paired over items)
    inter = {}
    for ctx in contexts:
        if ctx == "base" or ("base", "trait") not in ranges or (ctx, "trait") not in ranges:
            continue
        dlt = ranges[(ctx, "trait")] - ranges[("base", "trait")]
        inter[ctx] = {"range_diff": _f(np.nanmean(dlt)), "ci95": _ci(_nanmean(dlt[boot], 1))}
    out["interaction_vs_base"] = inter
    return out


def projections(acts_npz: Path, trait_npz: Path, layer: int, personas: list[str], contexts: list[str],
                mats: dict[str, np.ndarray] | None = None, which: str | None = "rpt") -> dict[str, Any]:
    """Trait-direction projection at the last prompt token before the decision, per persona and
    context (in units of the persona-centroid SD along the direction at base)."""
    A, meta = load_acts(acts_npz, layer)
    u = np.load(trait_npz)[f"L{layer}"].astype(np.float64)
    u /= np.linalg.norm(u)
    proj = A.astype(np.float64) @ u
    out: dict[str, Any] = {"layer": layer, "contexts": {}}
    contexts = [c for c in contexts if any(m["context"] == c and _is(m, which) for m in meta)]
    P = {c: np.array([proj[[k for k, m in enumerate(meta) if m["persona"] == p and m["context"] == c and _is(m, which)]].mean()
                      for p in personas]) for c in contexts}
    sd = float(P["base"].std(ddof=1))
    d_idx = personas.index("default")
    for c in contexts:
        e = {"default": float((P[c][d_idx] - P["base"][d_idx]) / sd), "mean_shift_vs_base": float((P[c] - P["base"]).mean() / sd),
             "per_persona": {p: float(P[c][i]) for i, p in enumerate(personas)}}
        if mats is not None and c in mats:
            from scipy.stats import spearmanr

            beh = _nanmean(mats[c], 1) if mats[c].ndim == 2 and mats[c].shape[1] > 12 else mats[c]
            if beh.ndim == 1:
                e["spearman_proj_vs_behaviour"] = float(spearmanr(P[c], beh).correlation)
            else:
                e["spearman_proj_vs_behaviour"] = {f"dim{j}": float(spearmanr(P[c], beh[:, j]).correlation)
                                                   for j in range(beh.shape[1])}
        out["contexts"][c] = e
    out["unit"] = "persona-centroid SD of the projection at base"
    return out


# ---------------------------------------------------------------------------
# dilemma panel (persona-choice items): profiles for the landscape; Track B on dilemmas
# ---------------------------------------------------------------------------

def fc_profiles(rows: list[dict], personas: list[str], contexts: list[str], split: str = "all") -> dict[str, np.ndarray]:
    """context -> [P, n_dims] pole_a rate per persona and dimension (mean over items of item means)."""
    dims = sorted({r["dimension"] for r in rows})
    out = {}
    for c in contexts:
        M = np.full((len(personas), len(dims)), np.nan)
        for i, p in enumerate(personas):
            for j, d in enumerate(dims):
                acc: dict[str, list] = defaultdict(list)
                for r in rows:
                    if (r["context"] == c and r["persona"] == p and r["dimension"] == d and r["valid"]
                            and (split == "all" or r["split"] == split)):
                        acc[r["item_id"]].append(r["pole_a"])
                if acc:
                    M[i, j] = np.mean([np.mean(v) for v in acc.values()])
        out[c] = M
    return out


def track_b_fc(rows_by: dict[tuple[str, str], list[dict]], choice: dict, contexts: list[str], n_random: int,
               seed: int = 0) -> dict[str, Any]:
    """Steering on dilemma items: pole_a rate (honesty: dishonest option; rules_outcomes: break the
    rule) per (dimension, persona), effect vs none, trait range +-s, random band, and the context
    interaction on the range. Items bootstrapped (shared resamples across cells)."""
    L, s = int(choice["layer"]), float(choice["strength"])
    fmt = lambda d, x: f"{d}@L{L}_{'+' if x >= 0 else '-'}{abs(x):g}"  # noqa: E731
    allrows = [r for rs in rows_by.values() for r in rs]
    dims = sorted({r["dimension"] for r in allrows})
    personas = sorted({r["persona"] for r in allrows}, key=lambda p: (p != "default", p))
    out: dict[str, Any] = {"layer": L, "strength": s, "by_dimension": {}}
    for d in dims:
        items = sorted({r["item_id"] for r in allrows if r["dimension"] == d})
        boot = np.random.default_rng(seed).integers(0, len(items), (N_BOOT, len(items)))

        def mat(ctx, cond):   # [P, I]
            M = np.full((len(personas), len(items)), np.nan)
            acc: dict[tuple, list] = defaultdict(list)
            for r in rows_by.get((ctx, cond), []):
                if r["dimension"] == d and r["valid"]:
                    acc[(r["persona"], r["item_id"])].append(r["pole_a"])
            for (p, it), v in acc.items():
                M[personas.index(p), items.index(it)] = np.mean(v)
            return M

        e_d: dict[str, Any] = {"contexts": {}}
        ranges = {}
        for ctx in contexts:
            none = mat(ctx, "none")
            e: dict[str, Any] = {"none": {p: _f(np.nanmean(none[i])) for i, p in enumerate(personas)}}
            for dname in ["trait"] + [f"random{k}" for k in range(n_random)] + ["traitdpo"]:
                P_, M_ = mat(ctx, fmt(dname, s)), mat(ctx, fmt(dname, -s))
                if np.all(np.isnan(P_)) or np.all(np.isnan(M_)):
                    continue
                r = P_ - M_                               # [P, I]
                pooled = _nanmean(r, 0)                   # [I] pooled over personas
                e[f"range_{dname}"] = {"pooled": _f(np.nanmean(pooled)), "pooled_ci95": _ci(_nanmean(pooled[boot], 1)),
                                       "by_persona": {p: _f(np.nanmean(r[i])) for i, p in enumerate(personas)}}
                ranges[(ctx, dname)] = pooled
                if dname == "trait":
                    for x in (s / 2, s, -s / 2, -s):
                        V = mat(ctx, fmt(dname, x))
                        e[fmt(dname, x)] = {"rate_pooled": _f(np.nanmean(V)),
                                            "diff_vs_none_pooled": _f(np.nanmean(V - none)),
                                            "by_persona": {p: _f(np.nanmean(V[i])) for i, p in enumerate(personas)}}
            band = [abs(e[f"range_random{k}"]["pooled"]) for k in range(n_random) if f"range_random{k}" in e]
            e["random_band"] = _f(max(band)) if band else None
            e_d["contexts"][ctx] = e
        inter = {}
        for ctx in contexts:
            if ctx != "base" and ("base", "trait") in ranges and (ctx, "trait") in ranges:
                dl = ranges[(ctx, "trait")] - ranges[("base", "trait")]
                inter[ctx] = {"range_diff": _f(np.nanmean(dl)), "ci95": _ci(_nanmean(dl[boot], 1))}
        e_d["interaction_vs_base"] = inter
        out["by_dimension"][d] = e_d
    out["personas"] = personas
    return out
