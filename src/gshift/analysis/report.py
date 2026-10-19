"""Q1 analysis for one model (plan sections "Primary analysis", "Test whether the shift is
shared", "Locate the shift in persona space", "Discovery and validation").

Confirmatory quantities are estimated on DISCOVERY roles x DISCOVERY questions only, per
question group (original, checkable) and pooled. Generalisation is then tested on held-out
roles, held-out questions and their intersection; PCA and the Assistant contrast are fit
on discovery data. All contrasts use matched units (valid in both conditions). A
full-dataset direction is saved too, labelled exploratory.

Writes outputs/<model>/analysis/{summary.json, tables/*.csv, vectors/*.npy|npz, figures/*.png}.
The primary inference is at the pre-registered layer for every representation; the sweep
is reported but never used to choose a layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_questions, role_categories
from ..design import build_requests
from ..io import model_dir, outputs_root, write_json
from . import stats as S
from .cells import Cells, build_cells, centroids, unit_normalised_direction
from .data import Store
from .space import persona_space_report

REP_NAMES = {"pre": "pre_response", "resp": "response_mean", "reason": "reasoning_mean"}


def _clean(obj: Any) -> Any:
    """JSON-safe copy (drops private keys, converts numpy, NaN -> None)."""
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


def score_cube(cells: Cells, meta: pd.DataFrame) -> np.ndarray:
    """Judge score aligned with cells [R, C, U] (NaN = missing)."""
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


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

def shared_block(mc: Cells, refs: dict[str, np.ndarray], cfg: dict, seed: int, full: bool,
                 balance_key: str) -> tuple[dict, np.ndarray]:
    """Everything in "Test whether the shift is shared" for one matched view."""
    D = S.disp(mc)
    g = D.mean(0)
    vnorm = {k: float(np.linalg.norm(v)) for k, v in refs.items()}
    obs = S.gram_stats(D @ D.T, Dv={k: D @ v for k, v in refs.items()}, vnorm=vnorm)
    n_perm = int(cfg["n_permutations"] if full else cfg.get("n_permutations_sweep", 500))
    n_boot = int(cfg["n_bootstrap"] if full else max(100, int(cfg["n_bootstrap"]) // 4))
    gu, near_zero = unit_normalised_direction(D)
    block = {
        "n_roles": len(mc.roles),
        "n_pairs_per_role": dict(zip(mc.roles, mc.n_pairs().astype(int).tolist())),
        "observed": obs,
        "null": S.summarise_null(obs, S.permutation_null(mc, n_perm, seed=seed, refs=refs)),
        "bootstrap": S.summarise_boot(S.bootstrap(mc, n_boot, seed=seed + 1, refs=refs)),
        "per_role": S.per_role_stats(D, mc.roles),
        "influence": S.influence_summary(D),
        "spectrum": S.spectrum(D),
        "residual_by_category": S.residual_category_structure(
            D, [mc.categories.get(r, "?") for r in mc.roles], seed=seed),
        "unit_normalised": {"cos_with_g": S.cos(g, gu), "excluded_near_zero_roles": [mc.roles[i] for i in near_zero],
                            "rule": "roles with ||Delta_r|| < 1e-3 x median excluded (direction undefined)"},
        "split_half": S.split_half_stability(mc, int(cfg.get("n_split_halves", 20)), seed=seed + 3,
                                             balance_key=balance_key),
        "family_loo": S.family_leave_one_out(mc),
        "crossfit_effect": S.crossfit_standardised_effect(mc),
        "g_norm": float(np.linalg.norm(g)),
    }
    for k, v in refs.items():
        block[f"cos_g_{k}"] = S.cos(g, v)
    return block, D


def generalisation_block(cg: Cells, a: str, b: str, g_disc: np.ndarray, disc_roles: list[str],
                         heldout: list[str], qsplit: np.ndarray, n_perm: int, seed: int) -> dict:
    out = {}
    views = {
        "heldout_roles_x_discovery_questions": (heldout, qsplit == "discovery"),
        "discovery_roles_x_validation_questions": (disc_roles, qsplit == "validation"),
        "heldout_roles_x_validation_questions": (heldout, qsplit == "validation"),
    }
    for name, (roles, umask) in views.items():
        if not roles or not umask.any():
            out[name] = {"skipped": "empty"}
            continue
        mc = cg.subset_units(umask).subset_roles(roles).matched(a, b)
        D = S.disp(mc)
        res = S.alignment_with(D, g_disc, mc.roles)
        res["null"] = S.alignment_null(mc, g_disc, n_perm, seed=seed)
        res["n_pairs"] = int(mc.n_pairs().sum())
        out[name] = res
    return out


def default_block(dcells: Cells, a: str, b: str, D_roles: np.ndarray, roles: list[str], umask: np.ndarray) -> dict:
    """Default-Assistant displacement, and each role's displacement relative to it."""
    if dcells.N.sum() == 0:
        return {"skipped": "no default-control data"}
    mc = dcells.subset_units(umask).matched(a, b)
    if mc.n_pairs().sum() == 0:
        return {"skipped": "no matched default pairs"}
    d = S.disp(mc)[0]
    g = D_roles.mean(0)
    per = []
    for r, dr in zip(roles, D_roles):
        nr = np.linalg.norm(dr)
        per.append({"role": r, "cos_with_default": S.cos(dr, d),
                    "norm_ratio_to_default": float(nr / np.linalg.norm(d)) if np.linalg.norm(d) > 0 else None,
                    # fraction of the role's shift NOT along the default's shift
                    "role_specific_frac": float(np.linalg.norm(dr - (dr @ d) / (d @ d) * d) ** 2 / nr ** 2)
                    if nr > 0 and d @ d > 0 else None})
    return {"n_pairs": int(mc.n_pairs().sum()), "norm": float(np.linalg.norm(d)), "cos_g_roles": S.cos(g, d),
            "mean_role_cos_with_default": float(np.nanmean([p["cos_with_default"] for p in per])),
            "per_role": per}


def adherence_filtered_block(cg: Cells, a: str, b: str, sc: np.ndarray, thr: float, min_pairs: int,
                             g_primary: np.ndarray) -> dict:
    """Secondary: matched pairs where BOTH responses meet the role-adherence criterion."""
    mask = (sc[:, cg.ci(a)] >= thr) & (sc[:, cg.ci(b)] >= thr)
    mc = cg.matched(a, b, extra_mask=mask)
    n = mc.n_pairs()
    keep = [r for r, k in zip(cg.roles, n) if k >= min_pairs]
    res: dict[str, Any] = {"threshold": thr, "min_pairs": min_pairs,
                           "n_pairs_per_role": dict(zip(cg.roles, n.astype(int).tolist())),
                           "insufficient_roles": [r for r, k in zip(cg.roles, n) if k < min_pairs]}
    if len(keep) < 3:
        res["skipped"] = "fewer than 3 roles with enough matched adherent pairs"
        return res
    D = S.disp(mc.subset_roles(keep))
    res["observed"] = S.gram_stats(D @ D.T)
    res["cos_with_primary_g"] = S.cos(D.mean(0), g_primary)
    res["n_roles"] = len(keep)
    return res


def adherence_report(meta: pd.DataFrame, contrasts: dict[str, list[str]], default_role: str) -> dict:
    from scipy import stats as st

    m = meta[meta["role"] != default_role]
    out: dict[str, Any] = {"by_condition": {}, "by_condition_and_group": {}}
    for c, sub in m.groupby("condition"):
        s = sub["score"].dropna()
        out["by_condition"][c] = {"n_scored": int(len(s)), "mean": float(s.mean()) if len(s) else None,
                                  "frac_fully_roleplaying": float((s == 3).mean()) if len(s) else None,
                                  "dist": {int(k): int(v) for k, v in s.value_counts().sort_index().items()}}
    for (c, g), sub in m.groupby(["condition", "question_group"]):
        s = sub["score"].dropna()
        out["by_condition_and_group"][f"{c}|{g}"] = {"n": int(len(s)), "mean": float(s.mean()) if len(s) else None}
    key = ["role", "prompt_variant", "question_id", "sample"]
    piv = m.pivot_table(index=key, columns="condition", values="score", aggfunc="first")
    for name, (a, b) in contrasts.items():
        if a not in piv or b not in piv:
            continue
        d = (piv[a] - piv[b]).dropna()
        res = {"n_pairs": int(len(d)), "mean_diff": float(d.mean()) if len(d) else None,
               "frac_up": float((d > 0).mean()) if len(d) else None,
               "frac_down": float((d < 0).mean()) if len(d) else None}
        nz = d[d != 0]
        if len(nz):
            res["sign_test_p"] = float(st.binomtest(int((nz > 0).sum()), len(nz)).pvalue)
        res["per_role_mean_diff"] = {r: float(v) for r, v in d.groupby(level="role").mean().items()}
        out[name] = res
    return out


def descriptives(meta: pd.DataFrame, exp: dict) -> dict:
    out: dict[str, Any] = {"by_condition": {}}
    for c, s in meta.groupby("condition"):
        out["by_condition"][c] = {
            "n": int(len(s)), "has_answer_rate": float(s["has_answer"].mean()),
            "truncation_rate": float(s["truncated"].mean()),
            "invalid_no_answer": int((~s["has_answer"]).sum()),
            "mean_answer_tokens": float(s["n_answer_tokens"].mean()),
            "mean_reasoning_tokens": float(s["n_reasoning_tokens"].mean()),
            "p95_generated_tokens": float(s["n_generated"].quantile(0.95))}
    have = set(meta["id"])   # missing = designed but not (yet) generated + extracted
    for stage in exp["stages"]:
        try:
            want = [r.id for r in build_requests(exp, stage)]
        except Exception:
            continue
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

def analyze_model(model_key: str, exp: dict, out_dir: Path | None = None,
                  reps: tuple[str, ...] = ("pre", "resp", "reason"), layers: list[int] | None = None,
                  groups: list[str] | None = None, make_plots: bool = True) -> dict:
    out_dir = out_dir or outputs_root() / model_key / "analysis"
    for sub in ("vectors", "tables", "figures"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    cfg = exp["analysis"]
    seed = int(cfg["seed"])
    store = Store.load(model_key)
    meta = store.meta
    cats = role_categories(exp)
    roles_all = list(cats)
    heldout = [r for r in exp["role_split"]["heldout"] if r in cats]
    disc_roles = [r for r in roles_all if r not in heldout]
    qmeta = {q["id"]: q for q in load_questions(exp)}
    contrasts = {k: list(v) for k, v in exp["contrasts"].items()}
    main_conds = list(exp["stages"]["main"]["conditions"])
    primary = int(exp["extraction"]["primary_layer"])
    layers = layers or store.layers
    groups = groups or list(cfg.get("question_groups", ["original", "checkable", "pooled"]))
    refs_by_layer = reference_vectors(model_key, layers)
    thr = float(exp["judge"]["role_expressed_threshold"])
    min_pairs = int(exp["judge"].get("min_matched_pairs_per_role", 50))
    droles = exp["default_role"]
    present = set(meta["condition"])
    para_conds = [c for c in exp["stages"].get("paraphrase", {}).get("conditions", []) if c in present]
    fol_conds = [c for c in exp["stages"].get("followup", {}).get("conditions", []) if c in present]

    summary: dict[str, Any] = {
        "model_key": model_key, "primary_layer": primary, "layers": layers, "groups": groups,
        "discovery_roles": disc_roles, "heldout_roles": heldout,
        "n_records": int(len(meta)), "descriptives": descriptives(meta, exp),
        "adherence": adherence_report(meta, contrasts, droles), "results": {},
        "notes": ["Confirmatory estimates use discovery roles x discovery questions; PCA and the Assistant "
                  "contrast are fit on discovery data. 'exploratory_full' uses all roles and questions.",
                  "Our Assistant contrast is a subset-based estimate (selected roles + default control), "
                  "not the original 275-role Assistant Axis.",
                  "Judge blinding is partial: the answer itself may reveal the condition."],
    }
    sweep_rows = []
    m_main = meta[meta["condition"].isin(main_conds) & meta["role"].isin(roles_all)].reset_index(drop=True)
    m_def = meta[meta["condition"].isin(main_conds) & (meta["role"] == droles)].reset_index(drop=True)
    units_main = sorted({(str(q), int(p), int(s)) for q, p, s in
                         zip(m_main["question_id"], m_main["prompt_variant"], m_main["sample"])})
    for rep in reps:
        summary["results"][rep] = {}
        for L in layers:
            full = L == primary
            refs = refs_by_layer.get(L, {})
            v = store.valid(rep, m_main)
            cells_all = build_cells(m_main, store.rows(rep, L, m_main), v, roles_all, main_conds, cats, qmeta,
                                    units_main)
            dX = store.rows(rep, L, m_def) if len(m_def) else np.zeros((0, cells_all.H), np.float32)
            dcells = build_cells(m_def, dX, store.valid(rep, m_def), [droles], main_conds, {}, qmeta, units_main)
            sc = score_cube(cells_all, m_main) if m_main["score"].notna().any() else None
            qgroup, qsplit = cells_all.unit_attr("group"), cells_all.unit_attr("split")
            base_rows = v & (m_main["condition"] == "baseline").to_numpy() & m_main["role"].isin(disc_roles).to_numpy()
            act_norm = float(store.norms(rep, L, m_main[base_rows]).mean()) if base_rows.any() else float("nan")
            res_L: dict[str, Any] = {"mean_activation_norm_baseline": act_norm}
            gs_disc_by_group: dict[str, dict[str, np.ndarray]] = {}
            disc_idx = [cells_all.ri(r) for r in disc_roles]
            for group in groups:
                gmask = np.ones(len(qgroup), bool) if group == "pooled" else qgroup == group
                if not gmask.any():
                    continue
                cg = cells_all.subset_units(gmask)
                qs_g = qsplit[gmask]
                dmask = qs_g == "discovery"
                c_disc = cg.subset_units(dmask).subset_roles(disc_roles)
                balance = "group" if group == "pooled" else "family"
                res_g: dict[str, Any] = {}
                gs_disc: dict[str, np.ndarray] = {}
                for name, (a, b) in contrasts.items():
                    mc = c_disc.matched(a, b)
                    block, D = shared_block(mc, refs, cfg, seed, full, balance)
                    g = D.mean(0)
                    gs_disc[name] = g
                    block["g_norm_over_act_norm"] = float(np.linalg.norm(g) / act_norm) if act_norm > 0 else None
                    block["generalisation"] = generalisation_block(
                        cg, a, b, g, disc_roles, heldout, qs_g,
                        n_perm=max(200, int(cfg["n_permutations"]) // (4 if full else 10)), seed=seed + 5)
                    block["default_assistant"] = default_block(dcells.subset_units(gmask), a, b, D, mc.roles, dmask)
                    if sc is not None:
                        block["adherence_filtered"] = adherence_filtered_block(
                            c_disc, a, b, sc[disc_idx][:, :, gmask][:, :, dmask], thr, min_pairs, g)
                    g_full = S.disp(cg.matched(a, b)).mean(0)
                    block["exploratory_full"] = {"cos_with_discovery_g": S.cos(g_full, g),
                                                 "g_norm": float(np.linalg.norm(g_full))}
                    stem = out_dir / "vectors" / f"{group}_{rep}_L{L}_{name}"
                    np.save(f"{stem}.npy", g.astype(np.float32))
                    np.save(f"{stem}_exploratory_full.npy", g_full.astype(np.float32))
                    n = np.linalg.norm(D, axis=1)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        pc = (D @ D.T) / np.outer(n, n)
                    np.savez(f"{stem}_role_displacements.npz", D=D.astype(np.float32), roles=np.array(mc.roles),
                             pairwise_cos=pc.astype(np.float32))
                    res_g[name] = block
                    sweep_rows.append(_sweep_row(group, rep, L, name, block, refs))
                # persona space: PCA fit on discovery-role baseline centroids (discovery questions)
                c_space = cg.subset_units(dmask)
                mu = centroids(c_space)
                mu_def = centroids(dcells.subset_units(gmask).subset_units(dmask))[0] if dcells.N.sum() > 0 else None
                space = persona_space_report(mu, main_conds, gs_disc, mu_def, list(cfg["pca_dims_report"]),
                                             fit_rows=disc_idx)
                if space.get("_axis") is not None:
                    np.save(out_dir / "vectors" / f"{group}_{rep}_L{L}_assistant_contrast.npy",
                            space["_axis"].astype(np.float32))
                np.save(out_dir / "vectors" / f"{group}_{rep}_L{L}_role_centroids.npy", mu.astype(np.float32))
                res_g["persona_space"] = space
                if make_plots and full:
                    from .plots import plot_layer_block

                    plot_layer_block(out_dir / "figures", f"{group}_{rep}", L, res_g, c_space, mu, mu_def,
                                     space, gs_disc, main_conds, heldout)
                gs_disc_by_group[group] = gs_disc
                res_L[group] = res_g
            if "original" in gs_disc_by_group and "checkable" in gs_disc_by_group:
                res_L["original_vs_checkable_cos"] = {
                    name: S.cos(gs_disc_by_group["original"][name], gs_disc_by_group["checkable"][name])
                    for name in contrasts}
            pooled = gs_disc_by_group.get("pooled") or next(iter(gs_disc_by_group.values()))
            if para_conds:
                res_L["paraphrases"] = paraphrase_block(store, meta, rep, L, roles_all, cats, qmeta, para_conds,
                                                        pooled, heldout)
            if fol_conds:
                res_L["followup"] = followup_block(store, meta, rep, L, cats, qmeta, fol_conds, pooled, disc_roles)
            summary["results"][rep][str(L)] = res_L
            first = list(contrasts)[0]
            msg = ", ".join(f"{gn}: |g|/act={res_L[gn][first]['g_norm_over_act_norm']:.4f} "
                            f"loo={res_L[gn][first]['observed']['mean_loo_cos']:.3f}" for gn in groups if gn in res_L)
            print(f"[analysis] {model_key} {rep} L{L} {first}: {msg}", flush=True)
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(out_dir / "tables" / "layer_sweep.csv", index=False)
    if make_plots and len(sweep):
        from .plots import plot_sweep

        plot_sweep(out_dir / "figures", sweep, primary)
    write_json(out_dir / "summary.json", _clean(summary))
    return summary


def _sweep_row(group: str, rep: str, L: int, name: str, block: dict, refs: dict) -> dict:
    gen = block["generalisation"]
    return {
        "group": group, "rep": rep, "layer": L, "contrast": name,
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
        "crossfit_d": block["crossfit_effect"]["mean_d"],
        **{f"cos_g_{k}": block.get(f"cos_g_{k}") for k in refs},
    }


def paraphrase_block(store: Store, meta: pd.DataFrame, rep: str, L: int, roles: list[str], cats: dict,
                     qmeta: dict, para_conds: list[str], gs: dict[str, np.ndarray], heldout: list[str]) -> dict:
    """Held-out grading wording on validation questions (prompt variant 0), matched against
    the main-stage baseline of the same units; compared with the discovery direction."""
    rows = meta[meta["role"].isin(roles) & (meta["prompt_variant"] == 0) & (meta["question_split"] == "validation")
                & meta["condition"].isin(["baseline"] + para_conds)].reset_index(drop=True)
    cells = build_cells(rows, store.rows(rep, L, rows), store.valid(rep, rows), roles,
                        ["baseline"] + para_conds, cats, qmeta)
    out: dict[str, Any] = {}
    held_idx = [cells.ri(r) for r in heldout if r in cells.roles]
    for c in para_conds:
        ref = "auto_minus_baseline" if c.startswith("auto") else "human_minus_baseline"
        other = "human_minus_baseline" if c.startswith("auto") else "auto_minus_baseline"
        D = S.disp(cells.matched(c, "baseline"))
        out[c] = {"cos_g_para_g_discovery": S.cos(D.mean(0), gs[ref]),
                  "mean_role_cos": float(np.nanmean([S.cos(d, gs[ref]) for d in D])),
                  # strictest: held-out roles x held-out questions x held-out wording
                  "cos_heldout_roles_g_discovery": S.cos(D[held_idx].mean(0), gs[ref]) if held_idx else None,
                  "cos_with_other_side": S.cos(D.mean(0), gs[other]),
                  "g_norm": float(np.linalg.norm(D.mean(0)))}
    autos = [c for c in para_conds if c.startswith("auto")]
    humans = [c for c in para_conds if c.startswith("human")]
    if autos and humans and "auto_minus_human" in gs:
        Da = np.mean([S.disp(cells.matched(c, "baseline")) for c in autos], 0)
        Dh = np.mean([S.disp(cells.matched(c, "baseline")) for c in humans], 0)
        out["auto_minus_human_pooled"] = {"cos_g_para_g_discovery": S.cos((Da - Dh).mean(0), gs["auto_minus_human"])}
    return out


def followup_block(store: Store, meta: pd.DataFrame, rep: str, L: int, cats: dict, qmeta: dict,
                   fol_conds: list[str], gs: dict[str, np.ndarray], disc_roles: list[str]) -> dict:
    """"This interaction is part of a model evaluation." vs matched baseline (prompt variant 0)."""
    rows = meta[meta["role"].isin(disc_roles) & (meta["prompt_variant"] == 0)
                & (meta["question_split"] == "discovery")
                & meta["condition"].isin(["baseline"] + fol_conds)].reset_index(drop=True)
    cells = build_cells(rows, store.rows(rep, L, rows), store.valid(rep, rows), disc_roles,
                        ["baseline"] + fol_conds, cats, qmeta)
    out = {}
    for c in fol_conds:
        D = S.disp(cells.matched(c, "baseline"))
        g = D.mean(0)
        out[c] = {"observed": S.gram_stats(D @ D.T), **{f"cos_g_{k}": S.cos(g, v) for k, v in gs.items()}}
    return out
