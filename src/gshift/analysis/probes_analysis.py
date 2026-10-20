"""Report on the context probes (CPU; memory: one condition's layer slice at a time, then
per-role means only). Inputs: outputs/<model>/probes/{persona,control}/ from
scripts/15_context_probes.py and Q1's outputs/<model>/analysis/vectors/.

Persona probe, per question group (original / checkable; Q1 analyses them separately) and
pooled, at the analysis layer, for every condition c vs baseline:
  D_r(c)        per-role shift = mean over the role's questions of x(c) - x(baseline)
                (prompts matched by role and question)
  shift_rel     ||mean_r D_r|| / mean baseline activation norm
  loo_cos       mean_r cos(D_r, mean_{s != r} D_s)  (is the shift shared across personas?)
  cos_q1        cos(mean shift, Q1 direction) for every Q1 direction in probes.yaml
  proj_t        D_r . unit(t), t = Q1 auto_minus_baseline of the matching group
  beta          cross-fitted isotropic contraction slope in the top-k Q1 baseline persona
                PCs: x_r = baseline position from one half of the questions (whole
                families), D_r from the other half; OLS D = a + beta x (one slope); both
                orientations of every family split, averaged. beta < 0 = personas move
                towards each other. Cross-fitting keeps noise in x_r from faking it.
  CIs           role bootstrap (cos with t, beta)
Control pairs: direction = mean(A) - mean(B) over the tasks; norm relative to the mean
activation norm; split-half reliability (tasks alternated within category); cosines with
the Q1 directions, the persona-probe shifts and each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..io import model_dir, read_jsonl
from ..probes import family_halves, probes_dir


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


# ---------------------------------------------------------------------------
# per-role means
# ---------------------------------------------------------------------------

class RoleMeans:
    """Per-role means of one condition within one question group (or pooled): `all`
    [R, H] over every question, and for every family split s the means over half A and
    half B (`halves[s] = (A, B)`); `norm` = mean per-prompt activation norm."""

    def __init__(self, roles: list[str], all_: np.ndarray, halves: list[tuple[np.ndarray, np.ndarray]],
                 norm: float, n_questions: int):
        self.roles, self.all, self.halves, self.norm, self.n_questions = roles, all_, halves, norm, n_questions


def role_means(X: np.ndarray, roles: np.ndarray, qids: np.ndarray, families: np.ndarray,
               role_order: list[str], splits: list[tuple[frozenset, frozenset]]) -> RoleMeans:
    """X [N, H] (one condition, one layer, the group's rows). Every role must have the same
    question set (matched design)."""
    X = np.asarray(X, dtype=np.float64)
    allm, halves = [], [([], []) for _ in splits]
    for r in role_order:
        m = roles == r
        if not m.any():
            raise ValueError(f"role {r!r} has no rows")
        allm.append(X[m].mean(0))
        for s, (fa, fb) in enumerate(splits):
            ma = m & np.isin(families, list(fa))
            mb = m & np.isin(families, list(fb))
            halves[s][0].append(X[ma].mean(0))
            halves[s][1].append(X[mb].mean(0))
    nq = len(set(qids[roles == role_order[0]]))
    return RoleMeans(list(role_order), np.stack(allm), [(np.stack(a), np.stack(b)) for a, b in halves],
                     float(np.linalg.norm(X, axis=1).mean()), nq)


# ---------------------------------------------------------------------------
# statistics (pure, on RoleMeans)
# ---------------------------------------------------------------------------

def loo_cos(D: np.ndarray) -> float:
    R = len(D)
    if R < 2:
        return float("nan")
    rest = (D.sum(0)[None] - D) / (R - 1)
    num = (D * rest).sum(1)
    den = np.linalg.norm(D, axis=1) * np.linalg.norm(rest, axis=1)
    return float(np.mean(np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)))


def _slope(x: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    """(sum xc.dc, sum xc.xc) of the isotropic OLS D = a + beta x on centred rows."""
    xc, dc = x - x.mean(0), d - d.mean(0)
    return float((xc * dc).sum()), float((xc * xc).sum())


def crossfit_parts(base: RoleMeans, cond: RoleMeans, pcs: np.ndarray, center: np.ndarray) -> list[tuple]:
    """[(x_k [R, k], D_k [R, k])] for every split and orientation: x from one half's
    baseline, D from the OTHER half's (cond - baseline), both projected on the PCs."""
    out = []
    for (bA, bB), (cA, cB) in zip(base.halves, cond.halves):
        for bx, by, cy in ((bA, bB, cB), (bB, bA, cA)):
            out.append(((bx - center) @ pcs.T, (cy - by) @ pcs.T))
    return out


def beta_from_parts(parts: list[tuple], idx: np.ndarray | None = None) -> float:
    vals = []
    for x, d in parts:
        if idx is not None:
            x, d = x[idx], d[idx]
        num, den = _slope(x, d)
        vals.append(num / den if den > 0 else np.nan)
    return float(np.nanmean(vals)) if vals else float("nan")


def shift_stats(base: RoleMeans, cond: RoleMeans, refs: dict[str, np.ndarray], t: np.ndarray | None,
                pcs: np.ndarray | None, center: np.ndarray | None, n_boot: int = 0, seed: int = 0,
                auto_ref: np.ndarray | None = None) -> dict[str, Any]:
    if base.roles != cond.roles:
        raise ValueError("role order differs between condition and baseline")
    D = cond.all - base.all
    g = D.mean(0)
    R = len(D)
    out: dict[str, Any] = {"n_roles": R, "n_questions": cond.n_questions,
                           "shift_norm": float(np.linalg.norm(g)),
                           "shift_rel": float(np.linalg.norm(g) / base.norm) if base.norm > 0 else float("nan"),
                           "mean_role_shift_norm_rel": float(np.linalg.norm(D, axis=1).mean() / base.norm),
                           "loo_cos": loo_cos(D),
                           "cos_q1": {k: cosine(g, v) for k, v in refs.items()}}
    if auto_ref is not None:
        out["cos_probe_auto_w1"] = cosine(g, auto_ref)
        out["norm_ratio_to_probe_auto_w1"] = float(np.linalg.norm(g) / np.linalg.norm(auto_ref)) \
            if np.linalg.norm(auto_ref) > 0 else float("nan")
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, R, R) for _ in range(n_boot)]
    if t is not None and np.linalg.norm(t) > 0:
        tu = _unit(t)
        p = D @ tu
        out["proj_t"] = {"mean": float(p.mean()), "sd": float(p.std(ddof=1)) if R > 1 else 0.0,
                         "min": float(p.min()), "max": float(p.max()), "frac_positive": float((p > 0).mean()),
                         "mean_over_shift_norm": float(p.mean() / np.linalg.norm(g)) if np.linalg.norm(g) > 0
                         else float("nan"),
                         "per_role": {r: round(float(x), 4) for r, x in zip(cond.roles, p)}}
        out["cos_t"] = cosine(g, t)
        if boots:
            G = D @ D.T
            bc = []
            for idx in boots:
                gn = np.sqrt(max(G[np.ix_(idx, idx)].sum(), 0)) / R
                bc.append(p[idx].mean() / gn if gn > 0 else np.nan)
            out["cos_t_ci95"] = [float(x) for x in np.nanpercentile(bc, [2.5, 97.5])]
    if pcs is not None and center is not None and base.halves:
        parts = crossfit_parts(base, cond, pcs, center)
        out["beta"] = beta_from_parts(parts)
        out["beta_n_fits"] = len(parts)
        if boots:
            out["beta_ci95"] = [float(x) for x in np.nanpercentile([beta_from_parts(parts, i) for i in boots],
                                                                   [2.5, 97.5])]
    return out


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def _layer_index(meta: dict, layer: int) -> int:
    layers = list(meta["layers"])
    if layer not in layers:
        raise ValueError(f"layer {layer} not captured (have {layers})")
    return layers.index(layer)


def _read_meta(d: Path) -> dict:
    import json

    return json.loads((d / "meta.json").read_text())


def load_slice(d: Path, name: str, li: int) -> tuple[np.ndarray, list[dict]]:
    """One file's rows at one layer as float64 [N, H] (memory-mapped read) + its index."""
    a = np.load(d / f"{name}.npy", mmap_mode="r")
    return np.asarray(a[:, li], dtype=np.float64), read_jsonl(d / f"{name}.index.jsonl")


def q1_vectors(model_key: str, names: list[str]) -> dict[str, np.ndarray]:
    vd = model_dir(model_key) / "analysis" / "vectors"
    return {n: np.load(vd / f"{n}.npy").astype(np.float64) for n in names if (vd / f"{n}.npy").exists()}


def q1_pcs(model_key: str, group: str, layer: int, k: int) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    vd = model_dir(model_key) / "analysis" / "vectors"
    p, c = vd / f"{group}_pre_L{layer}_baseline_pcs.npy", vd / f"{group}_pre_L{layer}_baseline_pca_center.npy"
    if not (p.exists() and c.exists()):
        return None, None
    return np.load(p)[:k].astype(np.float64), np.load(c).astype(np.float64)


def _group_means(X: np.ndarray, rows: list[dict], group: str | None, role_order: list[str]) -> RoleMeans:
    sel = np.array([group is None or r["group"] == group for r in rows])
    roles = np.array([r["role"] for r in rows])[sel]
    qids = np.array([r["question_id"] for r in rows])[sel]
    fams = np.array([r["family"] for r in rows])[sel]
    if group is None:   # pooled: split each group's families separately, union the halves
        groups = np.array([r["group"] for r in rows])[sel]
        per = [family_halves(fams[groups == g]) for g in dict.fromkeys(groups)]
        splits = [(frozenset().union(*[a for a, _ in c]), frozenset().union(*[b for _, b in c]))
                  for c in zip(*per)] if per and all(per) else []
    else:
        splits = family_halves(fams)
    return role_means(X[sel], roles, qids, fams, role_order, splits)


def persona_report(model_key: str, pcfg: dict) -> dict[str, Any]:
    d = probes_dir(model_key) / "persona"
    if not (d / "baseline.index.jsonl").exists():
        return {"skipped": "no persona probe outputs"}
    an = pcfg.get("analysis", {})
    L = int(pcfg.get("analysis_layer", 36))
    li = _layer_index(_read_meta(d), L)
    k, n_boot, seed = int(an.get("pca_k", 10)), int(an.get("n_bootstrap", 0)), int(an.get("seed", 0))
    refs = q1_vectors(model_key, list(an.get("q1_directions", [])))
    Xb, rb = load_slice(d, "baseline", li)
    keys_b = [(r["role"], r["question_id"]) for r in rb]
    role_order = list(dict.fromkeys(r["role"] for r in rb))
    groups = list(dict.fromkeys(r["group"] for r in rb))
    scopes = groups + ["pooled"]
    base = {s: _group_means(Xb, rb, None if s == "pooled" else s, role_order) for s in scopes}
    del Xb
    pcs = {g: q1_pcs(model_key, g, L, k) for g in scopes}
    tvec = {s: refs.get(f"{s}_pre_L{L}_auto_minus_baseline") for s in groups}
    pooled_t = model_dir(model_key) / "analysis" / "vectors" / f"pooled_pre_L{L}_auto_minus_baseline.npy"
    tvec["pooled"] = np.load(pooled_t).astype(np.float64) if pooled_t.exists() else None
    conds = [c for c in pcfg["persona_probe"]["conditions"] if c != "baseline" and (d / f"{c}.index.jsonl").exists()]
    # auto_w1 first: every other condition is also compared with the probe's own grading shift
    conds = sorted(conds, key=lambda c: c != "auto_w1")
    out: dict[str, Any] = {"layer": L, "pca_k": k, "n_bootstrap": n_boot, "n_roles": len(role_order),
                           "q1_directions_found": sorted(refs), "mean_baseline_norm": {s: base[s].norm for s in scopes},
                           "conditions": {}}
    auto_shift: dict[str, np.ndarray] = {}
    shifts: dict[str, dict[str, np.ndarray]] = {}
    for c in conds:
        Xc, rc = load_slice(d, c, li)
        if [(r["role"], r["question_id"]) for r in rc] != keys_b:
            raise RuntimeError(f"{c}: rows not matched to baseline (role, question)")
        res = {}
        shifts[c] = {}
        for s in scopes:
            cm = _group_means(Xc, rc, None if s == "pooled" else s, role_order)
            P, C = pcs[s]
            res[s] = shift_stats(base[s], cm, refs, tvec.get(s), P, C, n_boot, seed,
                                 auto_ref=auto_shift.get(s) if c != "auto_w1" else None)
            shifts[c][s] = (cm.all - base[s].all).mean(0)
            if c == "auto_w1":
                auto_shift[s] = shifts[c][s]
        del Xc
        tok = [r["n_prompt_tokens"] - b["n_prompt_tokens"] for r, b in zip(rc, rb)]
        res["extra_prompt_tokens_vs_baseline"] = {"mean": float(np.mean(tok)), "min": int(min(tok)),
                                                  "max": int(max(tok))}
        out["conditions"][c] = res
    out["_shifts"] = shifts            # popped by the caller (vectors, not JSON)
    return out


def _alternating_halves(rows: list[dict]) -> np.ndarray:
    """Mask of half A: tasks alternated within category (stratified split)."""
    seen: dict[str, int] = {}
    m = []
    for r in rows:
        c = r.get("task_category", "")
        m.append(seen.get(c, 0) % 2 == 0)
        seen[c] = seen.get(c, 0) + 1
    return np.array(m)


def control_report(model_key: str, pcfg: dict, persona_shifts: dict | None = None) -> dict[str, Any]:
    d = probes_dir(model_key) / "control"
    if not (d / "meta.json").exists():
        return {"skipped": "no control outputs"}
    an = pcfg.get("analysis", {})
    L = int(pcfg.get("analysis_layer", 36))
    li = _layer_index(_read_meta(d), L)
    refs = q1_vectors(model_key, list(an.get("q1_directions", [])))
    dirs, out = {}, {"layer": L, "pairs": {}}
    for pair in pcfg["control_pairs"]["pairs"]:
        if not ((d / f"{pair}_A.index.jsonl").exists() and (d / f"{pair}_B.index.jsonl").exists()):
            continue
        XA, rA = load_slice(d, f"{pair}_A", li)
        XB, rB = load_slice(d, f"{pair}_B", li)
        if [r["task_id"] for r in rA] != [r["task_id"] for r in rB]:
            raise RuntimeError(f"{pair}: A/B tasks not matched")
        diff = XA - XB
        v = diff.mean(0)
        h = _alternating_halves(rA)
        norm = float(np.concatenate([np.linalg.norm(XA, axis=1), np.linalg.norm(XB, axis=1)]).mean())
        dirs[pair] = v
        res = {"n_tasks": len(rA), "norm": float(np.linalg.norm(v)), "norm_rel": float(np.linalg.norm(v) / norm),
               "split_half_cos": cosine(diff[h].mean(0), diff[~h].mean(0)) if h.any() and (~h).any() else None,
               "loo_task_cos": loo_cos(diff),
               "cos_q1": {k: cosine(v, r) for k, r in refs.items()}}
        if persona_shifts:
            res["cos_persona_probe"] = {f"{c}|{s}": cosine(v, g) for c, by in persona_shifts.items()
                                        for s, g in by.items() if c in ("auto_w1", "human_w1", "neutral_long")}
        out["pairs"][pair] = res
    names = list(dirs)
    out["cos_between_pairs"] = {a: {b: cosine(dirs[a], dirs[b]) for b in names} for a in names}
    out["_dirs"] = dirs
    return out


# ---------------------------------------------------------------------------
# markdown
# ---------------------------------------------------------------------------

def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{nd}f}"


def _short(name: str) -> str:
    return name.replace("_pre_L36_", ":").replace("checkable", "chk").replace("original", "orig")


def markdown(rep: dict[str, Any]) -> str:
    lines = [f"# Context probes: {rep['model']}", ""]
    per = rep.get("persona", {})
    if "conditions" in per:
        lines += [f"Persona probe, layer {per['layer']}, {per['n_roles']} discovery roles, prompt variant 0. "
                  "shift_rel = ||mean shift|| / mean baseline norm; loo = mean leave-one-role-out cosine; "
                  "cos_t = cos(mean shift, Q1 auto_minus_baseline of the group) [role-bootstrap 95% CI]; "
                  "proj_t = mean role shift along unit(t); beta = cross-fitted contraction slope "
                  f"(top-{per['pca_k']} Q1 baseline PCs; < 0 = personas converge); "
                  "vs auto = cos / norm ratio with the probe's own auto_w1 shift.", ""]
        for s in list(per["mean_baseline_norm"]):
            lines += [f"## {s}", "",
                      "| condition | +tokens | shift_rel | loo | cos_t | proj_t | beta | cos vs auto | norm / auto |",
                      "|---|---|---|---|---|---|---|---|---|"]
            for c, res in per["conditions"].items():
                r = res[s]
                ci = r.get("cos_t_ci95")
                bci = r.get("beta_ci95")
                lines.append(
                    f"| {c} | {_f(res['extra_prompt_tokens_vs_baseline']['mean'], 1)} | {_f(r['shift_rel'])} | "
                    f"{_f(r['loo_cos'])} | {_f(r.get('cos_t'))}" + (f" [{_f(ci[0], 2)}, {_f(ci[1], 2)}]" if ci else "")
                    + f" | {_f(r.get('proj_t', {}).get('mean'), 2)} | {_f(r.get('beta'))}"
                    + (f" [{_f(bci[0], 2)}, {_f(bci[1], 2)}]" if bci else "")
                    + f" | {_f(r.get('cos_probe_auto_w1'))} | {_f(r.get('norm_ratio_to_probe_auto_w1'))} |")
            names = sorted({k for res in per["conditions"].values() for k in res[s]["cos_q1"]})
            if names:
                lines += ["", f"Cosine of the {s} mean shift with Q1 directions:", "",
                          "| condition | " + " | ".join(_short(n) for n in names) + " |",
                          "|---|" + "---|" * len(names)]
                for c, res in per["conditions"].items():
                    lines.append(f"| {c} | " + " | ".join(_f(res[s]["cos_q1"].get(n)) for n in names) + " |")
            lines.append("")
    con = rep.get("control", {})
    if con.get("pairs"):
        lines += [f"## Control pairs (layer {con['layer']}): direction = mean(A) - mean(B)", ""]
        names = sorted({k for r in con["pairs"].values() for k in r["cos_q1"]})
        lines += ["| pair | norm_rel | split-half cos | " + " | ".join(_short(n) for n in names) + " |",
                  "|---|---|---|" + "---|" * len(names)]
        for p, r in con["pairs"].items():
            lines.append(f"| {p} | {_f(r['norm_rel'])} | {_f(r['split_half_cos'])} | "
                         + " | ".join(_f(r["cos_q1"].get(n)) for n in names) + " |")
        pp = list(con["cos_between_pairs"])
        lines += ["", "Cosine between control directions:", "",
                  "| | " + " | ".join(str(i + 1) for i in range(len(pp))) + " |", "|---|" + "---|" * len(pp)]
        for i, a in enumerate(pp):
            lines.append(f"| {i + 1} {a} | " + " | ".join(_f(con["cos_between_pairs"][a][b], 2) for b in pp) + " |")
        lines.append("")
    return "\n".join(lines)


def probe_report(model_key: str, pcfg: dict) -> tuple[dict[str, Any], str]:
    per = persona_report(model_key, pcfg)
    shifts = per.pop("_shifts", None)
    con = control_report(model_key, pcfg, shifts)
    con.pop("_dirs", None)
    rep = {"model": model_key, "persona": per, "control": con}
    return rep, markdown(rep)
