"""Q1 analysis for one model.

Per representation x layer:
  * question groups analysed SEPARATELY (original: auto/human vs baseline; checkable:
    2 evaluators x 2 criteria vs baseline, with factorial evaluator / criterion /
    interaction contrasts); pooled contrasts are explicit, labelled averages of group
    contrasts (configs/experiment.yaml pooled_contrasts);
  * confirmatory estimates on DISCOVERY roles x DISCOVERY questions, matched units;
    generalisation to held-out roles, validation questions and their intersection, and to
    the reserved validation wordings;
  * "is it shared?" statistics with sign-flip nulls and roles x questions bootstraps,
    split-half and per-wording stability;
  * the three geometric hypotheses (geometry.py), cross-fitted by question family, at the
    pre-registered layer;
  * persona space: PCA fit on discovery-role baseline centroids (discovery questions);
    PC1 is only called Assistant-aligned if it actually aligns with our subset Assistant
    contrast (default control - mean role), reported with a role-bootstrap CI.
Writes outputs/<model>/analysis/{summary.json, tables/, vectors/, figures/}.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import heldout_roles, load_questions, role_categories
from ..design import build_requests, discovery_wording, question_ranks
from ..io import model_dir, outputs_root, write_json
from . import stats as S
from .cells import Cells, build_cells, centroids
from .contrast import ContrastData, bootstrap, disp, from_cells, permutation_null, pooled
from .data import Store
from .geometry import geometry_report
from .shared import alignment_null, crossfit_dz, family_leave_one_out, split_half_stability, wording_stability
from .space import captured_fraction, pairwise_distances, pca_fit, project

GROUPS = ("original", "checkable")


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items() if not str(k).startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return _clean(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def reference_vectors(model_key: str, layers: list[int]) -> dict[int, dict[str, np.ndarray]]:
    """Per layer: our reproduction of Betley's direction on this model, and (Qwen3.6 only)
    the published 0007 delta."""
    from ..betley import published

    out: dict[int, dict[str, np.ndarray]] = {L: {} for L in layers}
    repro = model_dir(model_key) / "betley_repro" / "deltas_all_layers.npy"
    if repro.exists():
        d = np.load(repro).astype(np.float64)
        for L in layers:
            if L < len(d) and np.linalg.norm(d[L]) > 0:
                out[L]["betley_repro"] = d[L]
    if model_key.startswith("qwen3.6"):
        pub = published()["deltas"].astype(np.float64)
        for L in layers:
            if L < len(pub) and np.linalg.norm(pub[L]) > 0:
                out[L]["betley_0007"] = pub[L]
    return out


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

def shared_block(cd: ContrastData, refs: dict[str, np.ndarray], cfg: dict, seed: int, full: bool,
                 balance_key: str) -> tuple[dict, np.ndarray]:
    D = disp(cd)
    g = D.mean(0)
    vnorm = {k: float(np.linalg.norm(v)) for k, v in refs.items()}
    obs = S.gram_stats(D @ D.T, Dv={k: D @ v for k, v in refs.items()}, vnorm=vnorm)
    n_perm = int(cfg["n_permutations"] if full else cfg.get("n_permutations_sweep", 500))
    n_boot = int(cfg["n_bootstrap"] if full else max(100, int(cfg["n_bootstrap"]) // 4))
    from .cells import unit_normalised_direction

    gu, near_zero = unit_normalised_direction(D)
    block = {
        "label": cd.label, "n_roles": len(cd.roles),
        "n_pairs_per_role": dict(zip(cd.roles, cd.n_pairs().astype(int).tolist())),
        "observed": obs,
        "null": S.summarise_null(obs, permutation_null(cd, n_perm, seed=seed, refs=refs)),
        "bootstrap": S.summarise_boot(bootstrap(cd, n_boot, seed=seed + 1, refs=refs)),
        "per_role": S.per_role_stats(D, cd.roles),
        "influence": S.influence_summary(D),
        "spectrum": S.spectrum(D),
        "residual_by_category": S.residual_category_structure(
            D, [cd.categories.get(r, "?") for r in cd.roles], n_perm=500, seed=seed),
        "unit_normalised": {"cos_with_g": S.cos(g, gu), "excluded_near_zero_roles": [cd.roles[i] for i in near_zero],
                            "rule": "roles with ||Delta_r|| < 1e-3 x median excluded (direction undefined)"},
        "split_half": split_half_stability(cd, int(cfg.get("n_split_halves", 20)), seed=seed + 3,
                                           balance_key=balance_key),
        "family_loo": family_leave_one_out(cd),
        "wording_stability": wording_stability(cd),
        "crossfit_effect": crossfit_dz(cd),
        "g_norm": float(np.linalg.norm(g)),
    }
    for k, v in refs.items():
        block[f"cos_g_{k}"] = S.cos(g, v)
    return block, D


def generalisation_block(cd_all: ContrastData, g_disc: np.ndarray, disc_roles: list[str], heldout: list[str],
                         n_perm: int, seed: int) -> dict:
    qsplit = cd_all.unit_attr("split")
    views = {
        "heldout_roles_x_discovery_questions": (heldout, qsplit == "discovery"),
        "discovery_roles_x_validation_questions": (disc_roles, qsplit == "validation"),
        "heldout_roles_x_validation_questions": (heldout, qsplit == "validation"),
    }
    out = {}
    for name, (roles, umask) in views.items():
        if not roles or not umask.any():
            out[name] = {"skipped": "empty"}
            continue
        cd = cd_all.subset_units(umask).subset_roles(roles)
        res = S.alignment_with(disp(cd), g_disc, cd.roles)
        res["null"] = alignment_null(cd, g_disc, n_perm, seed=seed)
        res["n_pairs"] = int(cd.n_pairs().sum())
        out[name] = res
    return out


def default_block(dcd: ContrastData | None, D_roles: np.ndarray, roles: list[str]) -> dict:
    """Default-Assistant displacement (analysed separately) and roles relative to it."""
    if dcd is None or dcd.M.sum() == 0:
        return {"skipped": "no default-control data"}
    d = disp(dcd)[0]
    g = D_roles.mean(0)
    per = []
    for r, dr in zip(roles, D_roles):
        nr = np.linalg.norm(dr)
        per.append({"role": r, "cos_with_default": S.cos(dr, d),
                    "norm_ratio_to_default": float(nr / np.linalg.norm(d)) if np.linalg.norm(d) > 0 else None,
                    "role_specific_frac": float(np.linalg.norm(dr - (dr @ d) / (d @ d) * d) ** 2 / nr ** 2)
                    if nr > 0 and d @ d > 0 else None})
    return {"n_units": int(dcd.M.sum()), "norm": float(np.linalg.norm(d)), "cos_g_roles": S.cos(g, d),
            "mean_role_cos_with_default": float(np.nanmean([p["cos_with_default"] for p in per])), "per_role": per}


def min_matched_pairs(exp: dict, group: str, split: str) -> int:
    """Secondary-analysis threshold for one question group and split (configs judge block:
    an int for every panel, or {group: {split: n}})."""
    v = exp["judge"].get("min_matched_pairs_per_role", 50)
    if isinstance(v, dict):
        v = v[group]
        if isinstance(v, dict):
            v = v[split]
    return int(v)


def role_sensitivity_block(cd: ContrastData, g: np.ndarray, sets: dict[str, list[str]]) -> dict:
    """The shared direction re-estimated without a set of discovery roles (e.g. grader-type
    roles, which may interact with the grading contrast)."""
    out = {}
    for name, drop in sets.items():
        keep = [r for r in cd.roles if r not in set(drop)]
        removed = [r for r in cd.roles if r in set(drop)]
        if len(keep) < 3:
            out[name] = {"skipped": "fewer than 3 roles left"}
            continue
        if not removed:
            out[name] = {"skipped": "none of these roles among the discovery roles"}
            continue
        D = disp(cd.subset_roles(keep))
        gs = D.mean(0)
        o = S.gram_stats(D @ D.T)
        out[name] = {"removed": removed, "n_roles": len(keep), "cos_with_g": S.cos(gs, g),
                     "norm_ratio": float(np.linalg.norm(gs) / np.linalg.norm(g)) if np.linalg.norm(g) > 0 else None,
                     "mean_loo_cos": o["mean_loo_cos"], "removed_roles_mean_cos_with_g_rest":
                         float(np.nanmean([S.cos(d, gs) for d in disp(cd.subset_roles(removed))]))}
    return out


def adherence_filtered_block(cd: ContrastData, cells: Cells, sc: np.ndarray, weights: dict, thr: float,
                             min_pairs: int, g_primary: np.ndarray) -> dict:
    """Secondary: units where EVERY involved condition's response meets the adherence criterion."""
    ok = np.ones(cd.M.shape, bool)
    for c, w in weights.items():
        if w != 0:
            ok &= sc[[cells.ri(r) for r in cd.roles]][:, cells.ci(c)][:, [cells.units.index(u) for u in cd.units]] >= thr
    M = cd.M & ok
    n = M.sum(1)
    res: dict[str, Any] = {"threshold": thr, "min_pairs": min_pairs,
                           "n_pairs_per_role": dict(zip(cd.roles, n.astype(int).tolist())),
                           "insufficient_roles": [r for r, k in zip(cd.roles, n) if k < min_pairs]}
    keep = [i for i, k in enumerate(n) if k >= min_pairs]
    if len(keep) < 3:
        res["skipped"] = "fewer than 3 roles with enough adherent matched units"
        return res
    f = ContrastData(cd.name, cd.label, [cd.roles[i] for i in keep], cd.units, cd.part, cd.part_weight,
                     cd.E[keep] * M[keep][..., None], M[keep], cd.qmeta, cd.categories, cd.wording)
    D = disp(f)
    res["observed"] = S.gram_stats(D @ D.T)
    res["cos_with_primary_g"] = S.cos(D.mean(0), g_primary)
    res["n_roles"] = len(keep)
    return res


def score_cube(cells: Cells, meta: pd.DataFrame) -> np.ndarray:
    sc = np.full(cells.N.shape, np.nan)
    ridx = {r: i for i, r in enumerate(cells.roles)}
    cidx = {c: i for i, c in enumerate(cells.conds)}
    uidx = {u: i for i, u in enumerate(cells.units)}
    for r, c, q, p, s, v in zip(meta["role"], meta["condition"], meta["question_id"], meta["prompt_variant"],
                                meta["sample"], meta["score"]):
        u = uidx.get((str(q), int(p), int(s)))
        if r in ridx and c in cidx and u is not None and v is not None and np.isfinite(v):
            sc[ridx[r], cidx[c], u] = v
    return sc


def persona_space(cells_by_group: dict[str, Cells], dcells_by_group: dict[str, Cells], group: str,
                  disc_roles: list[str], gs: dict[str, np.ndarray], cfg: dict, seed: int) -> dict:
    """PCA on discovery-role baseline centroids over discovery questions (pooled = mean of
    the two groups' baselines); Assistant contrast = default - mean(role) on the same data."""
    groups = list(GROUPS) if group == "pooled" else [group]
    mus, dmus = [], []
    for gname in groups:
        c = cells_by_group[gname]
        disc_u = np.array([c.qmeta.get(u[0], {}).get("split") == "discovery" for u in c.units])
        sub = c.subset_units(disc_u).subset_roles(disc_roles)
        mus.append(centroids(sub)[:, sub.ci("baseline")])
        dc = dcells_by_group.get(gname)
        if dc is not None and dc.N.sum() > 0:
            dmus.append(centroids(dc.subset_units(disc_u))[0, dc.ci("baseline")])
    B = np.mean(mus, 0)
    pca = pca_fit(B)
    ks = list(cfg["pca_dims_report"])
    rep: dict[str, Any] = {"pca_evr": pca["evr"][:10].tolist(), "n_components": int(len(pca["components"])),
                           "baseline_rms_spread": float(np.sqrt(((B - B.mean(0)) ** 2).sum(1).mean())),
                           "baseline_mean_pairwise_distance": float(pairwise_distances(B)[np.triu_indices(len(B), 1)].mean())
                           if len(B) > 1 else None}
    axis = np.mean(dmus, 0) - B.mean(0) if len(dmus) == len(groups) else None
    if axis is not None:
        pc1 = pca["components"][0]
        c = abs(S.cos(axis, pc1))
        rng = np.random.default_rng(seed)
        bs = []
        for _ in range(300):
            idx = rng.integers(0, len(B), len(B))
            if len(set(idx.tolist())) < 2:          # degenerate resample (tiny role sets): PC1 undefined
                continue
            p2 = pca_fit(B[idx])
            a2 = np.mean(dmus, 0) - B[idx].mean(0)
            bs.append(abs(S.cos(a2, p2["components"][0])))
        thr = float(cfg.get("assistant_alignment_threshold", 0.5))
        rep["assistant_contrast"] = {
            "norm": float(np.linalg.norm(axis)), "abs_cos_with_pc1": c,
            "abs_cos_with_pc1_ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))] if bs else None,
            "abs_cos_with_top5_pcs": [abs(S.cos(axis, v)) for v in pca["components"][:5]],
            "pc1_label": ("Assistant-aligned PC1" if c >= thr else "subset PC1 (NOT Assistant-aligned)"),
            "threshold": thr,
            "note": "subset-based Assistant contrast (our 40 discovery roles + default), not the original "
                    "275-role Assistant Axis"}
    for name, v in gs.items():
        r = {"captured_frac": captured_fraction(pca, v, ks),
             "random_direction_expectation": {k: min(k, len(pca["components"])) / v.size for k in ks},
             "cos_pc": [S.cos(v, pc) for pc in pca["components"][:5]]}
        if axis is not None:
            r["cos_assistant_contrast"] = S.cos(v, axis)
            ah = axis / np.linalg.norm(axis)
            sd = float((B @ ah).std())
            r["shift_along_assistant_over_role_sd"] = float(v @ ah / sd) if sd > 0 else None
        rep[name] = r
    rep["_pca"] = pca
    rep["_axis"] = axis
    rep["_B"] = B
    return rep


def adherence_report(meta: pd.DataFrame, exp: dict) -> dict:
    from scipy import stats as st

    m = meta[(meta["role"] != exp["default_role"]) & meta["wording_set"].isin(["discovery", "none"])]
    out: dict[str, Any] = {"by_group_condition": {}}
    for (g, c), sub in m.groupby(["question_group", "condition"]):
        s = sub["score"].dropna()
        out["by_group_condition"][f"{g}|{c}"] = {
            "n_scored": int(len(s)), "mean": float(s.mean()) if len(s) else None,
            "frac_fully_roleplaying": float((s == 3).mean()) if len(s) else None}
    key = ["role", "prompt_variant", "question_id", "sample"]
    for g in GROUPS:
        piv = m[m["question_group"] == g].pivot_table(index=key, columns="condition", values="score", aggfunc="first")
        for name, w in exp["contrasts"][g].items():
            if not all(c in piv for c in w):
                continue
            d = sum(float(wt) * piv[c] for c, wt in w.items()).dropna()
            res = {"n_units": int(len(d)), "mean": float(d.mean()) if len(d) else None}
            nz = d[d != 0]
            if len(nz):
                res["sign_test_p"] = float(st.binomtest(int((nz > 0).sum()), len(nz)).pvalue)
            out[f"{g}|{name}"] = res
    return out


def descriptives(meta: pd.DataFrame, exp: dict) -> dict:
    out: dict[str, Any] = {"by_group_condition": {}}
    for (g, c, w), s in meta.groupby(["question_group", "condition", "wording_set"]):
        out["by_group_condition"][f"{g}|{c}|{w}"] = {
            "n": int(len(s)), "has_answer_rate": float(s["has_answer"].mean()),
            "truncation_rate": float(s["truncated"].mean()), "invalid_no_answer": int((~s["has_answer"]).sum()),
            "mean_answer_tokens": float(s["n_answer_tokens"].mean()),
            "mean_reasoning_tokens": float(s["n_reasoning_tokens"].mean()),
            "p95_generated_tokens": float(s["n_generated"].quantile(0.95))}
    have = set(meta["id"])
    for stage in exp["stages"]:
        want = [r.id for r in build_requests(exp, stage)]
        miss = [i for i in want if i not in have]
        by_c: dict[str, int] = {}
        for i in miss:
            c = i.split("|")[3]
            by_c[c] = by_c.get(c, 0) + 1
        out[f"missing_{stage}"] = {"designed": len(want), "missing": len(miss), "by_condition": by_c}
    return out


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------

def _group_contrast(cells: Cells, dcells: Cells, name: str, w: dict, group: str, wording: np.ndarray) -> ContrastData:
    cd = from_cells(cells, name, w, group, wording=wording, keep_positions=True)
    cd._weights = w
    cd._default = from_cells(dcells, name, w, group, keep_positions=True)
    return cd


def _pooled_contrast(pname: str, pdef: dict, makers: dict) -> ContrastData:
    parts = {g: makers[(g, c)]() for g, c in pdef["parts"].items()}
    cd = pooled(pname, pdef["label"], parts)
    cd._default = pooled(pname, pdef["label"], {g: p._default for g, p in parts.items()})
    return cd


def _group_cells(store: Store, meta: pd.DataFrame, rep: str, L: int, group: str, roles: list[str], conds: list[str],
                 cats: dict, qmeta: dict, wording_filter=("discovery", "none"), H: int | None = None
                 ) -> tuple[Cells, pd.DataFrame]:
    m = meta[meta["role"].isin(roles) & (meta["question_group"] == group) & meta["condition"].isin(conds)
             & meta["wording_set"].isin(list(wording_filter))].reset_index(drop=True)
    units = sorted({(q["id"], p, 0) for q in qmeta.values() if q["group"] == group
                    for p in sorted(set(meta["prompt_variant"]))})
    cells = build_cells(m, store.rows(rep, L, m) if len(m) else np.zeros((0, H or 1), np.float32),
                        store.valid(rep, m), roles, conds, cats, qmeta, units)
    return cells, m


def analyze_model(model_key: str, exp: dict, out_dir: Path | None = None,
                  reps: tuple[str, ...] = ("pre", "resp", "reason"), layers: list[int] | None = None,
                  make_plots: bool = True, groups: list[str] | None = None) -> dict:
    out_dir = out_dir or outputs_root() / model_key / "analysis"
    for sub in ("vectors", "tables", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    cfg = exp["analysis"]
    seed = int(cfg["seed"])
    store = Store.load(model_key)
    meta = store.meta
    for col, default in (("wording_set", "none"), ("wording_id", "none")):
        if col not in meta:
            meta[col] = default
    cats = role_categories(exp)
    roles_all = list(cats)
    heldout = [r for r in heldout_roles(exp) if r in cats]
    disc_roles = [r for r in roles_all if r not in heldout]
    qrows = load_questions(exp)
    qmeta = {q["id"]: q for q in qrows}
    ranks = question_ranks(qrows)
    droles = exp["default_role"]
    primary = int(exp["extraction"]["primary_layer"])
    layers = layers or store.layers
    refs_by_layer = reference_vectors(model_key, layers)
    thr = float(exp["judge"]["role_expressed_threshold"])
    run_groups = [g for g in GROUPS if g in set(meta["question_group"])] if groups is None else \
        [g for g in groups if g in GROUPS]

    summary: dict[str, Any] = {
        "model_key": model_key, "primary_layer": primary, "layers": layers, "groups": run_groups,
        "discovery_roles": disc_roles, "heldout_roles": heldout, "n_records": int(len(meta)),
        "descriptives": descriptives(meta, exp), "adherence": adherence_report(meta, exp),
        "pooled_contrast_labels": {k: v["label"] for k, v in exp.get("pooled_contrasts", {}).items()},
        "results": {},
        "notes": ["Confirmatory estimates use discovery roles x discovery questions; PCA and the Assistant "
                  "contrast are fit on discovery data; 'exploratory_full' uses all roles and questions.",
                  "Original and checkable question groups are analysed separately; 'pooled' contrasts are the "
                  "labelled averages in pooled_contrast_labels.",
                  "A direction outside our sampled persona PCA subspace is not thereby outside all persona "
                  "representations.",
                  "Judge blinding is partial: the answer itself may reveal the condition."],
    }
    sweep_rows = []
    for rep in reps:
        summary["results"][rep] = {}
        for L in layers:
            full = L == primary
            refs = refs_by_layer.get(L, {})
            res_L: dict[str, Any] = {}
            # contrasts are built one at a time when analysed and freed after: each holds [R, U, H]
            # contrast vectors plus both state positions (~1-2 GB), so keeping all 17 alive at once
            # took ~20 GB per layer
            cells_g, dcells_g, metas_g, cds = {}, {}, {}, {}
            for group in run_groups:
                conds = list(exp["condition_sets"][group])
                cells_g[group], metas_g[group] = _group_cells(store, meta, rep, L, group, roles_all, conds, cats, qmeta)
                dcells_g[group], _ = _group_cells(store, meta, rep, L, group, [droles], conds, {}, qmeta,
                                                  H=cells_g[group].H)
                save_condition_centroids(out_dir, group, rep, L, cells_g[group], dcells_g[group])
                wording = np.array([discovery_wording(exp, ranks[u[0]], u[1]) for u in cells_g[group].units])
                for name, w in exp["contrasts"][group].items():
                    cds[(group, name)] = partial(_group_contrast, cells_g[group], dcells_g[group], name, w, group, wording)
            if set(GROUPS) <= set(run_groups):
                for pname, pdef in exp.get("pooled_contrasts", {}).items():
                    cds[("pooled", pname)] = partial(_pooled_contrast, pname, pdef, cds)
            base_m = meta[(meta["condition"] == "baseline") & meta["role"].isin(disc_roles)
                          & (meta["question_split"] == "discovery")]
            base_m = base_m[store.valid(rep, base_m)]
            act_norm = float(store.norms(rep, L, base_m).mean()) if len(base_m) else float("nan")
            res_L["mean_activation_norm_baseline"] = act_norm
            gs: dict[tuple[str, str], np.ndarray] = {}
            for (group, name), make_cd in cds.items():
                cd_all = make_cd()
                qs = cd_all.unit_attr("split")
                cd = cd_all.subset_units(qs == "discovery").subset_roles(disc_roles)
                block, D = shared_block(cd, refs, cfg, seed, full, "group" if group == "pooled" else "family")
                g = D.mean(0)
                gs[(group, name)] = g
                block["g_norm_over_act_norm"] = float(np.linalg.norm(g) / act_norm) if act_norm > 0 else None
                block["generalisation"] = generalisation_block(
                    cd_all, g, disc_roles, heldout, max(200, int(cfg["n_permutations"]) // (4 if full else 10)), seed + 5)
                dcd = getattr(cd_all, "_default", None)
                block["default_assistant"] = default_block(
                    None if dcd is None else dcd.subset_units(dcd.unit_attr("split") == "discovery"), D, cd.roles)
                if group != "pooled" and metas_g[group]["score"].notna().any():
                    block["adherence_filtered"] = adherence_filtered_block(
                        cd, cells_g[group], score_cube(cells_g[group], metas_g[group]), cd_all._weights, thr,
                        min_matched_pairs(exp, group, "discovery"), g)
                block["role_sensitivity"] = role_sensitivity_block(cd, g, cfg.get("sensitivity_role_sets", {}))
                g_full = disp(cd_all).mean(0)
                block["exploratory_full"] = {"cos_with_discovery_g": S.cos(g_full, g),
                                             "g_norm": float(np.linalg.norm(g_full))}
                if full:
                    vcd = cd_all.subset_units(qs == "validation").subset_roles(disc_roles)
                    hcd = cd_all.subset_units(qs == "discovery").subset_roles(heldout) if heldout else None
                    ddc = None if dcd is None else dcd.subset_units(dcd.unit_attr("split") == "discovery")
                    block["geometry"] = geometry_report(cd, cfg, seed=seed + 11, default_cd=ddc, heldout_cd=hcd,
                                                        validation_cd=vcd)
                stem = out_dir / "vectors" / f"{group}_{rep}_L{L}_{name}"
                np.save(f"{stem}.npy", g.astype(np.float32))
                np.save(f"{stem}_exploratory_full.npy", g_full.astype(np.float32))
                n = np.linalg.norm(D, axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    pc = (D @ D.T) / np.outer(n, n)
                np.savez(f"{stem}_role_displacements.npz", D=D.astype(np.float32), roles=np.array(cd.roles),
                         pairwise_cos=pc.astype(np.float32))
                res_L.setdefault(group, {})[name] = block
                sweep_rows.append(_sweep_row(group, rep, L, name, block, refs))
                cd_all = cd = dcd = D = vcd = hcd = ddc = None     # free before the next contrast is built
            # persona space per group and pooled
            for group in list(run_groups) + (["pooled"] if set(GROUPS) <= set(run_groups) else []):
                gsg = {name: v for (gr, name), v in gs.items() if gr == group}
                sp = persona_space(cells_g, dcells_g, group, disc_roles, gsg, cfg, seed)
                np.save(out_dir / "vectors" / f"{group}_{rep}_L{L}_baseline_pcs.npy", sp["_pca"]["components"].astype(np.float32))
                np.save(out_dir / "vectors" / f"{group}_{rep}_L{L}_baseline_pca_center.npy", sp["_pca"]["center"].astype(np.float32))
                if sp.get("_axis") is not None:
                    np.save(out_dir / "vectors" / f"{group}_{rep}_L{L}_assistant_contrast.npy", sp["_axis"].astype(np.float32))
                res_L.setdefault(group, {})["persona_space"] = sp
                if make_plots and full and group in cells_g:
                    _plots(out_dir, group, rep, L, cells_g[group], dcells_g[group], disc_roles, heldout, sp, exp)
            # do original and checkable directions agree?
            pairs = {"auto_minus_baseline": ("auto_minus_baseline", "auto_minus_baseline"),
                     "human_minus_baseline": ("human_minus_baseline", "human_minus_baseline"),
                     "automated_minus_human": ("auto_minus_human", "evaluator_main")}
            res_L["original_vs_checkable_cos"] = {
                k: S.cos(gs[("original", a)], gs[("checkable", b)]) for k, (a, b) in pairs.items()
                if ("original", a) in gs and ("checkable", b) in gs}
            res_L["validation_wordings"] = validation_wording_block(
                store, meta, rep, L, exp, roles_all, heldout, cats, qmeta, gs,
                n_perm=max(100, int(cfg["n_permutations"]) // (4 if full else 20)), seed=seed + 17)
            res_L["followup"] = followup_block(store, meta, rep, L, exp, disc_roles, cats, qmeta, gs)
            summary["results"][rep][str(L)] = res_L
            msg = ", ".join(f"{gr}/{nm}: |g|/act={b['g_norm_over_act_norm']:.4f} loo={b['observed']['mean_loo_cos']:.3f}"
                            for (gr, nm) in [("original", "auto_minus_baseline"), ("checkable", "evaluator_main")]
                            if (b := res_L.get(gr, {}).get(nm)))
            print(f"[analysis] {model_key} {rep} L{L}: {msg}", flush=True)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out_dir / "tables" / "layer_sweep.csv", index=False)
    if make_plots and len(sweep):
        from .plots import plot_sweep

        plot_sweep(out_dir / "figures", sweep, primary)
    write_json(out_dir / "summary.json", _clean(summary))
    return summary


def save_condition_centroids(out_dir: Path, group: str, rep: str, L: int, cells: Cells, dcells: Cells) -> None:
    """The persona representations themselves (not displacements): mu[r, c] for every role
    and condition, over discovery and validation questions separately, plus the default
    control. The graded persona is mu[r, graded]; Delta_r = mu[r, graded] - mu[r, baseline]."""
    out = {"roles": np.array(cells.roles), "conds": np.array(cells.conds)}
    for split in ("discovery", "validation"):
        keep = cells.unit_attr("split") == split
        if keep.any():
            out[f"mu_{split}"] = centroids(cells.subset_units(keep)).astype(np.float16)
            if dcells.N.sum() > 0:
                dk = dcells.unit_attr("split") == split
                out[f"mu_default_{split}"] = centroids(dcells.subset_units(dk))[0].astype(np.float16)
    np.savez(out_dir / "vectors" / f"{group}_{rep}_L{L}_condition_centroids.npz", **out)


def _sweep_row(group, rep, L, name, block, refs) -> dict:
    gen = block["generalisation"]
    geo = block.get("geometry", {}).get("observed", {})
    return {"group": group, "rep": rep, "layer": L, "contrast": name,
            "g_norm": block["g_norm"], "g_norm_over_act_norm": block["g_norm_over_act_norm"],
            "g_norm_debiased": block["null"].get("g_norm_debiased", {}).get("observed"),
            "g_norm_p": block["null"].get("g_norm", {}).get("p_greater"),
            "mean_loo_cos": block["observed"]["mean_loo_cos"],
            "loo_cos_null_q95": block["null"].get("mean_loo_cos", {}).get("null_q95"),
            "loo_p": block["null"].get("mean_loo_cos", {}).get("p_greater"),
            "mean_pairwise_cos": block["observed"]["mean_pairwise_cos"],
            "sv1_energy_frac": block["observed"]["sv1_energy_frac"],
            "split_half_cos": block["split_half"]["mean_cos"],
            "heldout_role_cos": gen.get("heldout_roles_x_discovery_questions", {}).get("cos_g"),
            "heldout_question_cos": gen.get("discovery_roles_x_validation_questions", {}).get("cos_g"),
            "heldout_both_cos": gen.get("heldout_roles_x_validation_questions", {}).get("cos_g"),
            "crossfit_dz": block["crossfit_effect"]["mean_dz"],
            "geometry_beta": geo.get("beta"), "geometry_separation_ratio": geo.get("separation_ratio"),
            **{f"cos_g_{k}": block.get(f"cos_g_{k}") for k in refs}}


def validation_wording_block(store, meta, rep, L, exp, roles, heldout, cats, qmeta, gs, n_perm: int = 200,
                             seed: int = 0) -> dict:
    """Reserved wordings (v1, v2) on validation questions, prompt variant 0, compared with
    the discovery direction of the same contrast.

    No baseline is regenerated for this panel: every graded record names its matched
    main-grid baseline record (baseline_id) and is paired with exactly that record. All
    wordings of a unit share ONE baseline record, so
      * per-wording contrasts ("<group>|<contrast>|v1", "...|v2") are NOT independent of
        each other (shared baseline noise) and are reported as such;
      * the combined estimate ("...|combined") averages the wordings' graded arms within a
        unit and subtracts the baseline once: one observation per unit, and its sign-flip
        null flips that single per-unit contrast;
      * the wording comparison ("...|v1_minus_v2") is graded@v1 - graded@v2, in which the
        shared baseline cancels exactly.
    """
    import itertools

    vm = meta[(meta["wording_set"] == "validation") & meta["role"].isin(roles)]
    if not len(vm):
        return {"skipped": "wording_validation stage not generated"}
    wids = list(exp["wording_templates"]["validation"])
    out: dict[str, Any] = {}
    for group in GROUPS:
        gv = vm[vm["question_group"] == group]
        if not len(gv):
            continue
        if "baseline_id" not in gv or gv["baseline_id"].isna().any():
            raise RuntimeError("wording_validation records without baseline_id: regenerate index with this code")
        base_ids = set(gv["baseline_id"])
        b = meta[meta["id"].isin(base_ids)]
        graded = gv.assign(condition=[f"{c}@{w}" for c, w in zip(gv["condition"], gv["wording_id"])])
        m = pd.concat([b, graded]).reset_index(drop=True)
        vconds = ["baseline"] + sorted(set(graded["condition"]))
        cells = build_cells(m, store.rows(rep, L, m), store.valid(rep, m), roles, vconds, cats, qmeta)
        out[f"{group}|records"] = {
            "n_graded_records": int(len(gv)), "n_baseline_records_referenced": len(base_ids),
            "n_baseline_records_missing": len(base_ids - set(b["id"])),
            "graded_records_per_baseline": float(len(gv) / max(len(base_ids), 1)),
            "note": "baselines are the main-grid records named by baseline_id, each counted once"}
        for name, w in exp["contrasts"][group].items():
            ref = gs.get((group, name))
            if ref is None:
                continue
            graded_w = {c: v for c, v in w.items() if c != "baseline"}

            def weights(wmap: dict[str, float]) -> dict[str, float]:
                return {k: v for k, v in wmap.items() if v != 0}

            def summarise(cd: ContrastData) -> dict:
                D = disp(cd)
                held = [cd.ri(r) for r in heldout if r in cd.roles]
                g = D.mean(0)
                return {"cos_g_wording_g_discovery": S.cos(g, ref),
                        "cos_heldout_roles_g_discovery": S.cos(D[held].mean(0), ref) if held else None,
                        "mean_role_cos": float(np.nanmean([S.cos(d, ref) for d in D])),
                        "norm_ratio": float(np.linalg.norm(g) / np.linalg.norm(ref)) if np.linalg.norm(ref) > 0 else None,
                        "n_units": int(cd.M.any(0).sum()), "_g": g}

            per = {}
            for wid in wids:
                ww = weights({(c if c == "baseline" else f"{c}@{wid}"): v for c, v in w.items()})
                if all(c in vconds for c in ww):
                    per[wid] = summarise(from_cells(cells, name, ww, group))
                    per[wid]["shares_baseline_with"] = [x for x in wids if x != wid]
                    out[f"{group}|{name}|{wid}"] = per[wid]
            if len(per) < 2:
                continue
            # combined: mean over wordings of the graded arms, baseline once per unit
            ww = weights({**({"baseline": w["baseline"]} if "baseline" in w else {}),
                          **{f"{c}@{wid}": v / len(wids) for c, v in graded_w.items() for wid in wids}})
            cd = from_cells(cells, name, ww, group)
            comb = summarise(cd)
            comb["null_mean_role_cos"] = alignment_null(cd, ref, n_perm, seed=seed)
            comb["baseline_reuse"] = "each unit's single baseline record enters once"
            out[f"{group}|{name}|combined"] = comb
            # wording differences: baseline cancels
            for a, bw in itertools.combinations(wids, 2):
                wd = weights({**{f"{c}@{a}": v for c, v in graded_w.items()},
                              **{f"{c}@{bw}": -v for c, v in graded_w.items()}})
                cdd = from_cells(cells, f"{name}:{a}-{bw}", wd, group)
                Dd = disp(cdd)
                obs = S.gram_stats(Dd @ Dd.T)
                nul = S.summarise_null(obs, permutation_null(cdd, n_perm, seed=seed + 1))
                gc = comb["_g"]
                out[f"{group}|{name}|{a}_minus_{bw}"] = {
                    "g_norm_over_combined_g_norm": float(obs["g_norm"] / np.linalg.norm(gc)) if np.linalg.norm(gc) > 0 else None,
                    "cos_with_g_discovery": S.cos(Dd.mean(0), ref),
                    "g_norm_p": nul.get("g_norm", {}).get("p_greater"),
                    "note": "graded@%s - graded@%s; the shared baseline cancels" % (a, bw)}
    return out


def followup_block(store, meta, rep, L, exp, disc_roles, cats, qmeta, gs) -> dict:
    """'This interaction is part of a model evaluation.' vs matched baseline (prompt variant 0)."""
    out = {}
    for group in GROUPS:
        m = meta[meta["role"].isin(disc_roles) & (meta["question_group"] == group) & (meta["prompt_variant"] == 0)
                 & (meta["question_split"] == "discovery") & meta["condition"].isin(["baseline", "eval_awareness"])
                 & meta["wording_set"].isin(["none", "followup"])].reset_index(drop=True)
        if not (m["condition"] == "eval_awareness").any():
            continue
        cells = build_cells(m, store.rows(rep, L, m), store.valid(rep, m), disc_roles, ["baseline", "eval_awareness"],
                            cats, qmeta)
        D = disp(from_cells(cells, "eval_awareness_minus_baseline", {"eval_awareness": 1, "baseline": -1}, group))
        g = D.mean(0)
        out[group] = {"observed": S.gram_stats(D @ D.T),
                      **{f"cos_g_{gr}|{nm}": S.cos(g, v) for (gr, nm), v in gs.items() if gr == group}}
    return out


def _plots(out_dir, group, rep, L, cells, dcells, disc_roles, heldout, sp, exp):
    from .plots import plot_arrows

    pca = sp["_pca"]
    disc_u = np.array([cells.qmeta.get(u[0], {}).get("split") == "discovery" for u in cells.units])
    sub = cells.subset_units(disc_u)
    mu = centroids(sub)
    ib = sub.ci("baseline")
    k = min(3, len(pca["components"]))
    Pb = project(pca, mu[:, ib], k)
    cats = [cells.categories.get(r, "?") for r in cells.roles]
    for c in sub.conds:
        if c == "baseline":
            continue
        Pc = project(pca, mu[:, sub.ci(c)], k)
        for dims in [d for d in ((0, 1), (0, 2)) if max(d) < k]:
            plot_arrows(out_dir / "figures" / f"arrows_{group}_{rep}_L{L}_{c}_pc{dims[0] + 1}{dims[1] + 1}.png",
                        Pb, Pc, cells.roles, cats,
                        f"{group} {rep} L{L}: baseline -> {c} (PCA on discovery roles; squares = held-out)",
                        dims, None, set(heldout))
