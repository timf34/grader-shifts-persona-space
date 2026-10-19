"""Qwen3.6-27B vs Qwen3.8-27B: is the grading-induced shift stronger, more consistent,
lower-dimensional, differently related to the Assistant contrast, or more behaviourally
consequential in the newer model?

Raw coordinates are NEVER compared across models (no assumption that the residual bases
align). Everything compared is either a within-model normalised scalar, or a
second-order (representational-similarity) quantity over the shared role set.
Model differences get a PAIRED bootstrap: the same resampled roles and questions are
used for both models in each replicate. Contrasts are built exactly as in report.py
(matched units, question groups separate, pooled = labelled average of group contrasts),
on discovery roles x discovery questions.
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
from .cells import centroids
from .contrast import ContrastData, disp, from_cells, pooled, question_weights, two_state
from .data import Store
from .space import pairwise_distances

SCALARS = [
    ("magnitude", "g_norm_over_act_norm"),
    ("magnitude", "crossfit_dz"),
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
BOOT_STATS = ("g_norm_over_act_norm", "mean_loo_cos", "mean_pairwise_cos", "sv1_energy_frac", "shared_energy_frac")
GROUPS = ("original", "checkable", "pooled")


def _summary(model_key: str) -> dict:
    p = outputs_root() / model_key / "analysis" / "summary.json"
    return json.loads(p.read_text())


def scalar_table(summ: dict) -> pd.DataFrame:
    rows = []
    for rep, by_layer in summ["results"].items():
        for L, res_L in by_layer.items():
            for group in GROUPS:
                res = res_L.get(group)
                if not isinstance(res, dict):
                    continue
                sp = res.get("persona_space", {})
                spread = sp.get("baseline_rms_spread")
                for name, b in res.items():
                    if name == "persona_space" or not isinstance(b, dict) or "observed" not in b:
                        continue
                    spc = sp.get(name, {})
                    cap = spc.get("captured_frac", {})
                    gen = b["generalisation"]
                    rows.append({
                        "rep": rep, "layer": int(L), "group": group, "contrast": name,
                        "g_norm_over_act_norm": b.get("g_norm_over_act_norm"),
                        "crossfit_dz": b["crossfit_effect"]["mean_dz"],
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
                        "cos_assistant_contrast": spc.get("cos_assistant_contrast"),
                        "captured_frac_k5": cap.get("5"),
                        "captured_frac_k20": cap.get("20"),
                        "loo_p": b["null"].get("mean_loo_cos", {}).get("p_greater"),
                        "g_norm_p": b["null"].get("g_norm", {}).get("p_greater"),
                    })
    return pd.DataFrame(rows)


def _model_contrasts(model_key: str, exp: dict, rep: str, L: int, disc_roles: list[str]) -> dict[str, Any]:
    """Discovery-role x discovery-question contrasts (as in report.py), condition centroids
    per contrast state (for RSA), and the baseline activation norm, for one model."""
    from .report import _group_cells

    store = Store.load(model_key)
    meta = store.meta
    for col, default in (("wording_set", "none"), ("wording_id", "none")):
        if col not in meta:
            meta[col] = default
    cats = role_categories(exp)
    qmeta = {q["id"]: q for q in load_questions(exp)}
    cds: dict[tuple[str, str], ContrastData] = {}
    states: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for group in ("original", "checkable"):
        if group not in set(meta["question_group"]):
            continue
        conds = list(exp["condition_sets"][group])
        cells, _ = _group_cells(store, meta, rep, L, group, disc_roles, conds, cats, qmeta)
        cells = cells.subset_units(cells.unit_attr("split") == "discovery")
        mu = centroids(cells)                                           # [R, C, H]
        for name, w in exp["contrasts"][group].items():
            cds[(group, name)] = from_cells(cells, name, w, group)
            ts = two_state(w)
            if ts is not None:
                states[(group, name)] = {k: sum(v * mu[:, cells.ci(c)] for c, v in side.items())
                                         for k, side in zip(("A", "B"), ts)}
        del cells
    for pname, pdef in exp.get("pooled_contrasts", {}).items():
        keys = [(g, c) for g, c in pdef["parts"].items()]
        if all(k in cds for k in keys):
            cds[("pooled", pname)] = pooled(pname, pdef["label"], {g: cds[(g, c)] for g, c in keys})
            if all(k in states for k in keys):
                states[("pooled", pname)] = {s: np.mean([states[k][s] for k in keys], 0) for s in ("A", "B")}
    base = meta[(meta["condition"] == "baseline") & meta["role"].isin(disc_roles)
                & (meta["question_split"] == "discovery") & meta["wording_set"].isin(["none"])]
    base = base[store.valid(rep, base)]
    return {"cds": cds, "states": states, "act_norm": float(store.norms(rep, L, base).mean())}


def second_order(model_a: str, model_b: str, rep: str, L: int, group: str, name: str,
                 states: dict[str, dict[str, np.ndarray]] | None = None) -> dict[str, Any]:
    """Representational similarity over the shared roles (no coordinate alignment assumed)."""
    out: dict[str, Any] = {}
    va = outputs_root() / model_a / "analysis" / "vectors"
    vb = outputs_root() / model_b / "analysis" / "vectors"
    stem = f"{group}_{rep}_L{L}_{name}_role_displacements.npz"
    fa, fb = va / stem, vb / stem
    if fa.exists() and fb.exists():
        za, zb = np.load(fa), np.load(fb)
        if list(za["roles"]) == list(zb["roles"]):
            Ca, Cb = za["pairwise_cos"].astype(np.float64), zb["pairwise_cos"].astype(np.float64)
            iu = np.triu_indices(len(Ca), 1)
            out["corr_displacement_cosine_matrices"] = float(np.corrcoef(Ca[iu], Cb[iu])[0, 1])
            na, nb = np.linalg.norm(za["D"], axis=1), np.linalg.norm(zb["D"], axis=1)
            out["corr_role_shift_norms"] = float(np.corrcoef(na, nb)[0, 1])
    if states is not None and model_a in states and model_b in states:
        sa, sb = states[model_a], states[model_b]
        iu = np.triu_indices(sa["B"].shape[0], 1)
        for s, label in (("B", "baseline"), ("A", "condition")):
            Da, Db = pairwise_distances(sa[s])[iu], pairwise_distances(sb[s])[iu]
            out[f"rsa_persona_distances_{label}"] = float(np.corrcoef(Da, Db)[0, 1])
        # do the two models change the same persona distances?
        cha = pairwise_distances(sa["A"])[iu] - pairwise_distances(sa["B"])[iu]
        chb = pairwise_distances(sb["A"])[iu] - pairwise_distances(sb["B"])[iu]
        out["corr_persona_distance_changes"] = float(np.corrcoef(cha, chb)[0, 1])
        out["state_labels"] = {"baseline": "contrast state B (reference)", "condition": "contrast state A"}
    return out


def paired_bootstrap(data: dict[str, dict], model_a: str, model_b: str, n_boot: int = 1000,
                     seed: int = 0) -> dict[tuple[str, str], dict[str, Any]]:
    """Model B minus model A per contrast, resampling the SAME roles and questions in both
    models in each replicate (discovery roles x discovery questions)."""
    rng = np.random.default_rng(seed)
    out: dict[tuple[str, str], dict[str, Any]] = {}
    ca, cb = data[model_a]["cds"], data[model_b]["cds"]
    for key in [k for k in ca if k in cb]:
        a, b = ca[key], cb[key]
        if a.units != b.units or a.roles != b.roles:
            raise RuntimeError(f"{key}: models have different unit/role layouts; paired bootstrap impossible")
        R = len(a.roles)
        diffs: dict[str, list[float]] = {k: [] for k in BOOT_STATS}
        for _ in range(n_boot):
            w = question_weights(rng, a)
            ridx = rng.integers(0, R, size=R)
            vals = {}
            for m, cd in ((model_a, a), (model_b, b)):
                D = disp(cd, w)[ridx]
                o = S.gram_stats(D @ D.T, origin=ridx)
                o["g_norm_over_act_norm"] = o["g_norm"] / data[m]["act_norm"]
                vals[m] = o
            for k in diffs:
                diffs[k].append(vals[model_b][k] - vals[model_a][k])
        res = {}
        for k, v in diffs.items():
            v = np.array(v, float)
            v = v[np.isfinite(v)]
            if len(v):
                res[k] = {"mean_diff": float(v.mean()), "lo": float(np.quantile(v, 0.025)),
                          "hi": float(np.quantile(v, 0.975)), "frac_positive": float((v > 0).mean())}
        out[key] = res
    return out


def compare_models(model_a: str, model_b: str, exp: dict, out_dir: Path | None = None,
                   n_boot: int = 1000, reps: tuple[str, ...] | None = None) -> dict:
    out_dir = out_dir or outputs_root() / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    ta = scalar_table(_summary(model_a)).assign(model=model_a)
    tb = scalar_table(_summary(model_b)).assign(model=model_b)
    both = pd.concat([ta, tb], ignore_index=True)
    both.to_csv(out_dir / "scalars_by_model.csv", index=False)
    L = int(exp["extraction"]["primary_layer"])
    cats = role_categories(exp)
    held = set(exp["role_split"]["heldout"])
    disc_roles = [r for r in cats if r not in held]
    res: dict[str, Any] = {"models": [model_a, model_b], "primary_layer": L, "primary": {},
                           "note": "discovery roles x discovery questions; paired bootstrap = same roles and "
                                   "questions resampled in both models; raw coordinates never compared"}
    for rep in reps or sorted(set(both["rep"])):
        res["primary"][rep] = {}
        try:
            data = {m: _model_contrasts(m, exp, rep, L, disc_roles) for m in (model_a, model_b)}
            boot = paired_bootstrap(data, model_a, model_b, n_boot, seed=int(exp["analysis"]["seed"]))
        except Exception as e:  # missing data for one model
            data, boot = None, {}
            res["primary"][rep]["skipped_paired_bootstrap"] = repr(e)
        for group in GROUPS:
            names = sorted(set(both[(both.rep == rep) & (both.group == group)]["contrast"]))
            for name in names:
                rows = both[(both.rep == rep) & (both.layer == L) & (both.contrast == name) & (both.group == group)]
                if len(rows) != 2:
                    continue
                ra = rows[rows.model == model_a].iloc[0].to_dict()
                rb = rows[rows.model == model_b].iloc[0].to_dict()
                states = None
                if data is not None and all((group, name) in data[m]["states"] for m in data):
                    states = {m: data[m]["states"][(group, name)] for m in data}
                res["primary"][rep][f"{group}|{name}"] = {
                    "by_group": {grp: {k: {model_a: ra.get(k), model_b: rb.get(k)} for g2, k in SCALARS if g2 == grp}
                                 for grp in dict.fromkeys(g for g, _ in SCALARS)},
                    "second_order": second_order(model_a, model_b, rep, L, group, name, states),
                    "paired_bootstrap_b_minus_a": boot.get((group, name), {"skipped": "no paired data"}),
                }
    beh = {}
    for m in (model_a, model_b):
        p = outputs_root() / m / "steering" / "behaviour_summary.json"
        if p.exists():
            beh[m] = json.loads(p.read_text())
    res["behaviour"] = beh
    res["caveat"] = ("A stronger effect in the newer model establishes a difference between releases; "
                     "attributing it to additional RLVR needs training information or controlled checkpoints.")
    from .report import _clean

    write_json(out_dir / "comparison.json", _clean(res))
    return res
