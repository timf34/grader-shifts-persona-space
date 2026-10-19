"""Qwen3.6-27B vs Qwen3.8-27B: is the grading-induced shift stronger, more consistent,
lower-dimensional, differently related to the Assistant Axis, or more behaviourally
consequential in the newer model?

Raw coordinates are NEVER compared across models (no assumption that the residual bases
align). Everything compared is either a within-model normalised scalar, or a
second-order (representational-similarity) quantity over the shared role set.
Model differences get a PAIRED bootstrap: the same resampled roles and questions are
used for both models in each replicate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_questions, role_categories
from ..io import outputs_root, write_json
from . import stats as S
from .cells import build_cells, centroids
from .data import Store
from .space import pairwise_distances

SCALARS = [
    ("magnitude", "g_norm_over_act_norm"),
    ("magnitude", "crossfit_d"),
    ("magnitude", "g_over_persona_spread"),
    ("consistency", "mean_loo_cos"),
    ("consistency", "mean_pairwise_cos"),
    ("consistency", "split_half_cos"),
    ("generalisation", "heldout_role_cos"),
    ("generalisation", "heldout_question_cos"),
    ("generalisation", "heldout_both_cos"),
    ("dimensionality", "sv1_energy_frac"),
    ("dimensionality", "participation_ratio"),
    ("dimensionality", "residual_energy_frac"),
    ("assistant_contrast", "cos_assistant_contrast"),
    ("assistant_contrast", "captured_frac_k5"),
    ("assistant_contrast", "captured_frac_k20"),
]


def _summary(model_key: str) -> dict:
    p = outputs_root() / model_key / "analysis" / "summary.json"
    return json.loads(p.read_text())


def scalar_table(summ: dict) -> pd.DataFrame:
    rows = []
    for rep, by_layer in summ["results"].items():
        for L, res_L in by_layer.items():
            for group, res in res_L.items():
                if not isinstance(res, dict) or "persona_space" not in res:
                    continue
                sp = res["persona_space"]
                for name in ("auto_minus_baseline", "human_minus_baseline", "auto_minus_human"):
                    b = res.get(name)
                    if not b:
                        continue
                    spc = sp.get(name, {})
                    cap = spc.get("captured_frac", {})
                    spread = sp.get("distances_auto_vs_baseline", {}).get("rms_spread_baseline")
                    gen = b["generalisation"]
                    rows.append({
                        "rep": rep, "layer": int(L), "group": group, "contrast": name,
                        "g_norm_over_act_norm": b.get("g_norm_over_act_norm"),
                        "crossfit_d": b["crossfit_effect"]["mean_d"],
                        "g_over_persona_spread": (b["observed"]["g_norm"] / spread) if spread else None,
                        "mean_loo_cos": b["observed"]["mean_loo_cos"],
                        "mean_pairwise_cos": b["observed"]["mean_pairwise_cos"],
                        "split_half_cos": b["split_half"]["mean_cos"],
                        "heldout_role_cos": gen.get("heldout_roles_x_discovery_questions", {}).get("cos_g"),
                        "heldout_question_cos": gen.get("discovery_roles_x_validation_questions", {}).get("cos_g"),
                        "heldout_both_cos": gen.get("heldout_roles_x_validation_questions", {}).get("cos_g"),
                        "sv1_energy_frac": b["observed"]["sv1_energy_frac"],
                        "participation_ratio": b["observed"]["participation_ratio"],
                        "residual_energy_frac": b["observed"]["residual_energy_frac"],
                        "cos_assistant_contrast": spc.get("cos_assistant_axis"),
                        "captured_frac_k5": cap.get("5"),
                        "captured_frac_k20": cap.get("20"),
                        "loo_p": b["null"].get("mean_loo_cos", {}).get("p_greater"),
                        "g_norm_p": b["null"].get("g_norm", {}).get("p_greater"),
                    })
    return pd.DataFrame(rows)


def second_order(model_a: str, model_b: str, rep: str, L: int, group: str, name: str) -> dict[str, Any]:
    """Representational similarity over the shared roles (no coordinate alignment assumed)."""
    out: dict[str, Any] = {}
    va = outputs_root() / model_a / "analysis" / "vectors"
    vb = outputs_root() / model_b / "analysis" / "vectors"
    fa, fb = va / f"{group}_{rep}_L{L}_{name}_role_displacements.npz", vb / f"{group}_{rep}_L{L}_{name}_role_displacements.npz"
    if fa.exists() and fb.exists():
        za, zb = np.load(fa), np.load(fb)
        if list(za["roles"]) == list(zb["roles"]):
            Ca, Cb = za["pairwise_cos"].astype(np.float64), zb["pairwise_cos"].astype(np.float64)
            iu = np.triu_indices(len(Ca), 1)
            out["corr_displacement_cosine_matrices"] = float(np.corrcoef(Ca[iu], Cb[iu])[0, 1])
            na, nb = np.linalg.norm(za["D"], axis=1), np.linalg.norm(zb["D"], axis=1)
            out["corr_role_shift_norms"] = float(np.corrcoef(na, nb)[0, 1])
    ca, cb = va / f"{group}_{rep}_L{L}_role_centroids.npy", vb / f"{group}_{rep}_L{L}_role_centroids.npy"
    if ca.exists() and cb.exists():
        mua, mub = np.load(ca).astype(np.float64), np.load(cb).astype(np.float64)   # [R, C, H]
        iu = np.triu_indices(mua.shape[0], 1)
        for ci, c in enumerate(("baseline", "auto", "human")):
            if ci < mua.shape[1]:
                Da, Db = pairwise_distances(mua[:, ci]), pairwise_distances(mub[:, ci])
                out[f"rsa_persona_distances_{c}"] = float(np.corrcoef(Da[iu], Db[iu])[0, 1])
    return out


def paired_bootstrap(model_a: str, model_b: str, exp: dict, rep: str, L: int, n_boot: int = 1000,
                     seed: int = 0) -> dict[str, Any]:
    """Model B minus model A (discovery roles x discovery questions, pooled), resampling the
    SAME roles and questions in both models in each replicate."""
    cats = role_categories(exp)
    heldout = set(exp["role_split"]["heldout"])
    disc_roles = [r for r in cats if r not in heldout]
    qmeta = {q["id"]: q for q in load_questions(exp)}
    conds = list(exp["stages"]["main"]["conditions"])
    cells, norms = {}, {}
    for m in (model_a, model_b):
        st = Store.load(m)
        meta = st.meta[st.meta["condition"].isin(conds) & st.meta["role"].isin(disc_roles)
                       & (st.meta["question_split"] == "discovery")].reset_index(drop=True)
        v = st.valid(rep, meta)
        cells[m] = build_cells(meta, st.rows(rep, L, meta), v, disc_roles, conds, cats, qmeta)
        base = v & (meta["condition"] == "baseline").to_numpy()
        norms[m] = float(st.norms(rep, L, meta[base]).mean())
    if cells[model_a].units != cells[model_b].units:
        raise RuntimeError("models have different unit layouts; paired bootstrap impossible")
    rng = np.random.default_rng(seed)
    uq = cells[model_a].unit_question()
    R = len(disc_roles)
    out: dict[str, Any] = {}
    for name, (a, b) in exp["contrasts"].items():
        mcs = {m: cells[m].matched(a, b) for m in cells}
        diffs: dict[str, list[float]] = {"g_norm_over_act_norm": [], "mean_loo_cos": [], "sv1_energy_frac": [],
                                         "mean_pairwise_cos": []}
        for _ in range(n_boot):
            w = S.question_weights(rng, uq)
            ridx = rng.integers(0, R, size=R)
            vals = {}
            for m, mc in mcs.items():
                mu = centroids(mc, w)
                D = np.nan_to_num(mu[ridx, 0] - mu[ridx, 1])
                o = S.gram_stats(D @ D.T, origin=ridx)
                o["g_norm_over_act_norm"] = o["g_norm"] / norms[m]
                vals[m] = o
            for k in diffs:
                diffs[k].append(vals[model_b][k] - vals[model_a][k])
        out[name] = {k: {"mean_diff": float(np.mean(v)), "lo": float(np.quantile(v, 0.025)),
                         "hi": float(np.quantile(v, 0.975)), "frac_positive": float(np.mean(np.array(v) > 0))}
                     for k, v in diffs.items()}
    return out


def compare_models(model_a: str, model_b: str, exp: dict, out_dir: Path | None = None,
                   n_boot: int = 1000) -> dict:
    out_dir = out_dir or outputs_root() / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    ta = scalar_table(_summary(model_a)).assign(model=model_a)
    tb = scalar_table(_summary(model_b)).assign(model=model_b)
    both = pd.concat([ta, tb], ignore_index=True)
    both.to_csv(out_dir / "scalars_by_model.csv", index=False)
    L = int(exp["extraction"]["primary_layer"])
    res: dict[str, Any] = {"models": [model_a, model_b], "primary_layer": L, "primary": {}}
    for rep in sorted(set(both["rep"])):
        res["primary"][rep] = {}
        for name in exp["contrasts"]:
            rows = both[(both.rep == rep) & (both.layer == L) & (both.contrast == name) & (both.group == "pooled")]
            if len(rows) != 2:
                continue
            ra = rows[rows.model == model_a].iloc[0].to_dict()
            rb = rows[rows.model == model_b].iloc[0].to_dict()
            res["primary"][rep][name] = {
                "by_group": {grp: {k: {model_a: ra.get(k), model_b: rb.get(k)} for g2, k in SCALARS if g2 == grp}
                             for grp in dict.fromkeys(g for g, _ in SCALARS)},
                "second_order": second_order(model_a, model_b, rep, L, "pooled", name),
            }
        try:
            res["primary"][rep]["paired_bootstrap_b_minus_a"] = paired_bootstrap(model_a, model_b, exp, rep, L, n_boot)
        except Exception as e:  # missing data for one model
            res["primary"][rep]["paired_bootstrap_b_minus_a"] = {"skipped": repr(e)}
    beh = {}
    for m in (model_a, model_b):
        p = outputs_root() / m / "steering" / "behaviour_summary.json"
        if p.exists():
            beh[m] = json.loads(p.read_text())
    res["behaviour"] = beh
    res["caveat"] = ("A stronger effect in the newer model establishes a difference between releases; "
                     "attributing it to additional RLVR needs training information or controlled checkpoints.")
    write_json(out_dir / "comparison.json", res)
    return res
