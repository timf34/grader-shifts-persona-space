#!/usr/bin/env python3
"""How does RL change the persona landscape, and the effect of grading context?

Offline (CPU) comparison of three checkpoints of ONE RLVR trajectory of OLMo-3 32B Think:
olmo3-32b-dpo (before RLVR) -> olmo31-32b-rl100 (step 100, ~4% of the run) ->
olmo31-32b-final (full RLVR), from each checkpoint's saved analysis files only:

    outputs/<model>/analysis/summary.json             (activation norms, saved values to reproduce)
    outputs/<model>/analysis/vectors/*_role_displacements.npz, *_condition_centroids.npz
    outputs/<model>/<condition>/scores/<role>.jsonl   (judge role-adherence scores 0-3)

Never loads outputs/<model>/<cond>/activations/*. Peak memory well under 1 GB.

    GSHIFT_EXPERIMENT=configs/experiment_olmo_rl.yaml python scripts/olmo_rl_compare.py

Writes outputs/olmo_rl_compare/{report.json, report.md, figures/*.png}.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("GSHIFT_EXPERIMENT", "configs/experiment_olmo_rl.yaml")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gshift.analysis import rl_compare as rc  # noqa: E402
from gshift.analysis.contrast import two_state  # noqa: E402
from gshift.config import heldout_roles, load_experiment, load_questions, role_categories  # noqa: E402
from gshift.io import outputs_root  # noqa: E402

MODELS = ["olmo3-32b-dpo", "olmo31-32b-rl100", "olmo31-32b-final"]
MLABEL = {"olmo3-32b-dpo": "DPO (pre-RL)", "olmo31-32b-rl100": "RL step 100", "olmo31-32b-final": "RL final"}
SHORT = {"olmo3-32b-dpo": "dpo", "olmo31-32b-rl100": "rl100", "olmo31-32b-final": "final"}
PAIRS = [("olmo3-32b-dpo", "olmo31-32b-rl100"), ("olmo3-32b-dpo", "olmo31-32b-final"),
         ("olmo31-32b-rl100", "olmo31-32b-final")]
PKEY = {p: f"{SHORT[p[1]]}-{SHORT[p[0]]}" for p in PAIRS}           # "rl100-dpo", "final-dpo", "final-rl100"
REPS = ["pre", "resp", "reason"]
LAYERS = [16, 24, 32, 36, 40, 48, 56]
PRIMARY = 36
CONTRASTS = {"original": ["auto_minus_baseline", "neutral_minus_baseline", "auto_minus_neutral", "auto_minus_human"],
             "checkable": ["auto_correct_minus_baseline", "neutral_minus_baseline", "auto_correct_minus_neutral",
                           "evaluator_main", "criterion_main", "betley_matched"]}
GROUPS = ["original", "checkable"]
LGROUPS = ["original", "checkable", "pooled"]
N_BOOT = 2000
JACK_METS = ("pr", "pc1_share", "abs_cos_pc1_assistant")
SEED = 20261023

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MCOL = dict(zip(MODELS, ["#52514e", "#eda100", "#2a78d6"]))
RCOL = dict(zip(REPS, SERIES[:3]))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
CATS = ["analytical_helping", "relational", "competitive", "creative", "nonhuman"]
CAT_LABEL = {"analytical_helping": "analytical / helping", "relational": "relational", "competitive": "competitive",
             "creative": "creative", "nonhuman": "nonhuman"}
CAT_MARK = {"analytical_helping": "o", "relational": "s", "competitive": "^", "creative": "D", "nonhuman": "v"}
REP_LABEL = {"pre": "pre (last prompt token)", "resp": "resp (final answer)", "reason": "reason (reasoning)"}
CLABEL = {("original", "auto_minus_baseline"): "open: auto − baseline",
          ("original", "neutral_minus_baseline"): "open: neutral − baseline",
          ("original", "auto_minus_neutral"): "open: auto − neutral",
          ("original", "auto_minus_human"): "open: auto − human",
          ("checkable", "auto_correct_minus_baseline"): "chk: auto_correct − baseline",
          ("checkable", "neutral_minus_baseline"): "chk: neutral_long − baseline",
          ("checkable", "auto_correct_minus_neutral"): "chk: auto_correct − neutral_long",
          ("checkable", "evaluator_main"): "chk: evaluator main (auto − human)",
          ("checkable", "criterion_main"): "chk: criterion main (correct − quality)",
          ("checkable", "betley_matched"): "chk: auto_correct − human_quality"}


def peak_gb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1e9 if sys.platform == "darwin" else r / 1e6


def adir(m: str) -> Path:
    return outputs_root() / m / "analysis"


def vec(m: str, name: str) -> Path:
    return adir(m) / "vectors" / name


def r6(x):
    if isinstance(x, dict):
        return {k: r6(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [r6(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else round(float(x), 6)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_meta(exp: dict) -> dict:
    """Per checkpoint: activation norms per rep/layer and the saved values we reproduce.
    summary.json is ~27 MB: loaded one checkpoint at a time and freed."""
    meta = {"act_norm": {}, "saved": {}, "saved_adherence": {}, "disc_roles": None}
    for m in MODELS:
        s = json.load(open(adir(m) / "summary.json"))
        if meta["disc_roles"] is None:
            meta["disc_roles"] = list(s["discovery_roles"])
        elif list(s["discovery_roles"]) != meta["disc_roles"]:
            raise ValueError(f"{m}: discovery roles differ from {MODELS[0]}")
        if sorted(s["heldout_roles"]) != sorted(heldout_roles(exp)):
            raise ValueError(f"{m}: held-out roles differ from the config")
        meta["act_norm"][m] = {rep: {L: float(s["results"][rep][str(L)]["mean_activation_norm_baseline"])
                                     for L in LAYERS} for rep in REPS}
        sv = {}
        for rep in REPS:
            for L in LAYERS:
                for g, names in CONTRASTS.items():
                    for nm in names:
                        b = s["results"][rep][str(L)][g][nm]
                        sv[(rep, L, g, nm)] = {
                            "g_norm": b["g_norm"], "g_norm_over_act_norm": b["g_norm_over_act_norm"],
                            "mean_loo_cos": b["observed"]["mean_loo_cos"],
                            "separation_ratio_crossfit": b.get("geometry", {}).get("observed", {}).get("separation_ratio"),
                            "beta_crossfit": b.get("geometry", {}).get("observed", {}).get("beta")}
                for g in LGROUPS:
                    ps = s["results"][rep][str(L)][g]["persona_space"]
                    sv[(rep, L, g, "persona_space")] = {
                        "pca_evr": ps["pca_evr"], "baseline_rms_spread": ps["baseline_rms_spread"],
                        "abs_cos_pc1_assistant": ps.get("assistant_contrast", {}).get("abs_cos_with_pc1")}
        meta["saved"][m] = sv
        meta["saved_adherence"][m] = {k: v for k, v in s["adherence"].items() if k != "by_group_condition"}
        meta["saved_adherence"][m]["by_group_condition"] = s["adherence"]["by_group_condition"]
        del s
    return meta


def load_centroids(m: str, group: str, rep: str, L: int) -> dict:
    z = np.load(vec(m, f"{group}_{rep}_L{L}_condition_centroids.npz"))
    return {"roles": [str(r) for r in z["roles"]], "conds": [str(c) for c in z["conds"]],
            **{k: z[k].astype(np.float64) for k in ("mu_discovery", "mu_validation",
                                                     "mu_default_discovery", "mu_default_validation")}}


def baseline_landscape_inputs(m: str, group: str, rep: str, L: int) -> dict:
    """Baseline centroids of all 50 roles + default, discovery and validation questions;
    'pooled' = mean of the two groups' baselines (as in report.persona_space)."""
    parts = [load_centroids(m, g, rep, L) for g in (GROUPS if group == "pooled" else [group])]
    roles = parts[0]["roles"]
    for p in parts:
        if p["roles"] != roles:
            raise ValueError("role order differs between groups")
    out = {"roles": roles}
    for split in ("discovery", "validation"):
        out[f"B_{split}"] = np.mean([p[f"mu_{split}"][:, p["conds"].index("baseline")] for p in parts], 0)
        out[f"d_{split}"] = np.mean([p[f"mu_default_{split}"][p["conds"].index("baseline")] for p in parts], 0)
    return out


# ---------------------------------------------------------------------------
# 1. grading shifts: size, LOO alignment, separation
# ---------------------------------------------------------------------------

def state_positions(cen: dict, weights: dict, roles: list[str], split: str) -> tuple[np.ndarray, np.ndarray]:
    ts = two_state(weights)
    if ts is None:
        raise ValueError(f"not a two-state contrast: {weights}")
    ri = [cen["roles"].index(r) for r in roles]
    mu = cen[f"mu_{split}"][ri]
    pos = []
    for side in ts:
        P = np.zeros((len(ri), mu.shape[-1]))
        for c, w in side.items():
            P += w * mu[:, cen["conds"].index(c)]
        pos.append(P)
    return pos[0], pos[1]


def shifts_section(exp: dict, meta: dict, idx: np.ndarray) -> dict:
    disc = meta["disc_roles"]
    R = len(disc)
    I = rc.ident(R)
    out: dict = {}
    for rep in REPS:
        for L in LAYERS:
            full = L == PRIMARY
            cens = {(m, g): load_centroids(m, g, rep, L) for m in MODELS for g in GROUPS} if full else {}
            for g, names in CONTRASTS.items():
                for nm in names:
                    per, boot, gvec = {}, {}, {}
                    for m in MODELS:
                        z = np.load(vec(m, f"{g}_{rep}_L{L}_{nm}_role_displacements.npz"))
                        if [str(r) for r in z["roles"]] != disc:
                            raise ValueError(f"{m} {g} {nm}: D roles differ from discovery roles")
                        D = z["D"].astype(np.float64)
                        gvec[m] = D.mean(0)
                        G = D @ D.T
                        act = meta["act_norm"][m][rep][L]
                        gn0, loo0 = rc.shift_stats(G, I)
                        gnb, loob = rc.shift_stats(G, idx)
                        st = {"size": (gn0[0] / act, gnb / act), "loo": (loo0[0], loob)}
                        if full:
                            A, B = state_positions(cens[(m, g)], exp["contrasts"][g][nm], disc, "discovery")
                            Av, Bv = state_positions(cens[(m, g)], exp["contrasts"][g][nm], disc, "validation")
                            SA, SB = rc.pair_cross(A), rc.pair_cross(B)
                            st["sep"] = ((rc.mean_pair(SA, I) / rc.mean_pair(SB, I))[0],
                                         rc.mean_pair(SA, idx) / rc.mean_pair(SB, idx))
                            # noise-robust variant: discovery x validation cross-products
                            # (independent question sets: question noise cancels in expectation)
                            XA, XB = rc.pair_cross(A, Av), rc.pair_cross(B, Bv)
                            st["sep_xsplit"] = ((rc.mean_pair(XA, I) / rc.mean_pair(XB, I))[0],
                                                rc.mean_pair(XA, idx) / rc.mean_pair(XB, idx))
                        boot[m] = {k: v[1] for k, v in st.items()}
                        per[m] = {k: rc.summarise(*v) for k, v in st.items()}
                        sv = meta["saved"][m][(rep, L, g, nm)]
                        per[m]["saved"] = {"g_norm_over_act_norm": sv["g_norm_over_act_norm"],
                                           "mean_loo_cos": sv["mean_loo_cos"],
                                           **({"separation_ratio_crossfit": sv["separation_ratio_crossfit"]}
                                              if full else {})}
                    diffs = {PKEY[(a, b)]: {k: rc.diff_summary(per[a][k]["est"], per[b][k]["est"], boot[a][k], boot[b][k])
                                            for k in boot[a]} for a, b in PAIRS}
                    cosg = {PKEY[(a, b)]: float(gvec[a] @ gvec[b] / np.linalg.norm(gvec[a]) / np.linalg.norm(gvec[b]))
                            for a, b in PAIRS}
                    out.setdefault(rep, {}).setdefault(str(L), {}).setdefault(g, {})[nm] = {
                        "per_checkpoint": per, "diffs": diffs, "raw_cos_g_across_checkpoints": cosg}
            print(f"[shifts] {rep} L{L} done  (peak {peak_gb():.2f} GB)", flush=True)
    return out


# ---------------------------------------------------------------------------
# 2. adherence
# ---------------------------------------------------------------------------

def load_scores(exp: dict, m: str, qmeta: dict) -> dict:
    """{(group, condition): {(role, question): score}} for the main (discovery-wording) grid,
    prompt variant 0, sample 0, prompted roles only; the last row per id wins."""
    disc_words = set(exp["wording_templates"]["discovery"])
    conds = sorted({c for g in GROUPS for c in exp["condition_sets"][g]})
    roles = set(role_categories(exp))
    rows: dict[str, dict] = {}
    for c in conds:
        for f in sorted((outputs_root() / m / c / "scores").glob("*.jsonl")):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    rows[r["id"]] = r
    out: dict = defaultdict(dict)
    skipped = defaultdict(int)
    for i, r in rows.items():
        role, pv, q, c, w, smp = i.split("|")
        if role not in roles:
            skipped["non-prompted role"] += 1
            continue
        if pv != "p0" or smp != "s0" or not (w in disc_words or w == "none"):
            skipped["other wording/variant/sample"] += 1
            continue
        s = r.get("score")
        if s is None or not np.isfinite(float(s)):
            skipped["no score"] += 1
            continue
        g = qmeta[q]["group"]
        if c not in exp["condition_sets"][g]:
            skipped["condition not in group"] += 1
            continue
        out[(g, c)][(role, q)] = float(s)
    return {"scores": dict(out), "skipped": dict(skipped)}


def load_lengths(exp: dict, m: str, qmeta: dict) -> dict:
    """Answer / reasoning token counts per (group, condition) and (role, question) from the
    response records (streamed line by line; only the counts are kept)."""
    disc_words = set(exp["wording_templates"]["discovery"])
    roles = set(role_categories(exp))
    out: dict = {"answer": defaultdict(dict), "reasoning": defaultdict(dict)}
    for c in sorted({c for g in GROUPS for c in exp["condition_sets"][g]}):
        for f in sorted((outputs_root() / m / c / "responses").glob("*.jsonl")):
            for line in open(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                role, pv, q, cc, w, smp = r["id"].split("|")
                if role not in roles or pv != "p0" or smp != "s0" or not (w in disc_words or w == "none"):
                    continue
                g = qmeta[q]["group"]
                if cc != c or c not in exp["condition_sets"][g] or not r.get("has_answer", True):
                    continue
                out["answer"][(g, c)][(role, q)] = float(r["n_answer_tokens"])
                out["reasoning"][(g, c)][(role, q)] = float(r["n_reasoning_tokens"])
    return {k: dict(v) for k, v in out.items()}


def adherence_section(exp: dict, qmeta: dict) -> tuple[dict, dict]:
    scores = {m: load_scores(exp, m, qmeta) for m in MODELS}
    out = paired_section(exp, {m: scores[m]["scores"] for m in MODELS}, SEED + 101)
    out["skipped_rows"] = {m: scores[m]["skipped"] for m in MODELS}
    return out, scores


def lengths_section(exp: dict, qmeta: dict) -> tuple[dict, dict]:
    """Paired log token-count ratios (condition vs reference) per checkpoint and across
    checkpoints; exp(mean log ratio) = geometric-mean length ratio."""
    lens = {m: load_lengths(exp, m, qmeta) for m in MODELS}
    out = {}
    for kind in ("answer", "reasoning"):
        logv = {m: {k: {u: float(np.log(max(x, 1.0))) for u, x in d.items()} for k, d in lens[m][kind].items()}
                for m in MODELS}
        out[kind] = paired_section(exp, logv, SEED + 202 + (kind == "reasoning"))
        out[kind]["raw_mean_tokens"] = {f"{g}|{c}": {SHORT[m]: float(np.mean(list(lens[m][kind][(g, c)].values())))
                                                     for m in MODELS}
                                        for g in GROUPS for c in exp["condition_sets"][g]}
    return out, lens


def length_link(exp: dict, meta: dict, lens: dict, idx: np.ndarray) -> dict:
    """Across the 40 discovery personas: does a persona's representation shift along the
    shared direction track how much its answers shorten? Per checkpoint, Pearson r between
    (D_r . g_hat) / act and the persona's mean paired log answer-length ratio (state A vs B)."""
    disc = meta["disc_roles"]
    out: dict = {}
    keys = [("original", "auto_minus_baseline"), ("original", "auto_minus_neutral"),
            ("checkable", "auto_correct_minus_baseline"), ("checkable", "auto_correct_minus_neutral"),
            ("checkable", "evaluator_main")]
    for rep in ("resp", "reason"):
        for g, nm in keys:
            A, B = two_state(exp["contrasts"][g][nm])
            res = {}
            for m in MODELS:
                D = np.load(vec(m, f"{g}_{rep}_L{PRIMARY}_{nm}_role_displacements.npz"))["D"].astype(np.float64)
                gh = D.mean(0) / np.linalg.norm(D.mean(0))
                proj = D @ gh / meta["act_norm"][m][rep][PRIMARY]
                ans = lens[m]["answer"]
                lr = []
                for r in disc:
                    qs = set.intersection(*[{q for (rr, q) in ans[(g, c)] if rr == r} for c in list(A) + list(B)])
                    v = [sum(w * np.log(max(ans[(g, c)][(r, q)], 1)) for c, w in A.items())
                         - sum(w * np.log(max(ans[(g, c)][(r, q)], 1)) for c, w in B.items()) for q in qs]
                    lr.append(np.mean(v) if v else np.nan)
                lr = np.array(lr)
                ok = np.isfinite(lr)
                rb = np.array([np.corrcoef(proj[ix][ok[ix]], lr[ix][ok[ix]])[0, 1] for ix in idx[:1000]])
                res[SHORT[m]] = {**rc.summarise(np.corrcoef(proj[ok], lr[ok])[0, 1], rb),
                                 "mean_log_len_ratio": float(np.nanmean(lr)),
                                 "sd_log_len_ratio_across_personas": float(np.nanstd(lr)),
                                 "mean_proj_over_act": float(proj.mean())}
            out.setdefault(rep, {})[f"{g}|{nm}"] = res
    return out


def paired_section(exp: dict, vals: dict, seed: int) -> dict:
    """vals[model][(group, condition)][(role, question)] = value. Per checkpoint: condition
    means and paired condition - reference differences (reference = baseline, and the
    length-matched neutral for graded conditions); across checkpoints: the difference of
    those differences, paired by (role, question). CIs: bootstrap over (role, question)
    units, and a two-way (roles x questions) bootstrap."""
    held = set(heldout_roles(exp))
    allr = list(role_categories(exp))
    subsets = {"all": set(allr), "discovery": set(allr) - held, "heldout": held}
    scores = {m: {"scores": vals[m]} for m in MODELS}
    out: dict = {"groups": {}}
    for g in GROUPS:
        conds = list(exp["condition_sets"][g])
        neutral = "neutral" if g == "original" else "neutral_long"
        graded = [c for c in conds if c not in ("baseline", neutral)]
        comps = [(c, "baseline") for c in conds if c != "baseline"] + [(c, neutral) for c in graded]
        qs = sorted({q for (r, q) in scores[MODELS[0]]["scores"][(g, "baseline")]})
        G: dict = {}
        for sname, rset in subsets.items():
            rl = sorted(rset)
            blk: dict = {"means": {}, "diffs": {}, "did": {}}
            unitdiff: dict = {}
            for m in MODELS:
                sc = scores[m]["scores"]
                for c in conds:
                    v = np.array([s for (r, q), s in sc[(g, c)].items() if r in rset])
                    seed += 1
                    blk["means"].setdefault(c, {})[SHORT[m]] = {**rc.summarise(v.mean(), rc.unit_boot_mean(v, N_BOOT, seed)),
                                                                 "n": int(len(v))}
                for c, ref in comps:
                    d = {u: sc[(g, c)][u] - sc[(g, ref)][u] for u in sc[(g, c)]
                         if u in sc[(g, ref)] and u[0] in rset}
                    unitdiff[(m, c, ref)] = d
                    v = np.array(list(d.values()))
                    V = np.full((len(rl), len(qs)), np.nan)
                    for (r, q), x in d.items():
                        V[rl.index(r), qs.index(q)] = x
                    seed += 1
                    blk["diffs"].setdefault(f"{c}-{ref}", {})[SHORT[m]] = {
                        **rc.summarise(v.mean(), rc.unit_boot_mean(v, N_BOOT, seed)),
                        "ci_twoway": rc.ci(rc.twoway_boot_mean(V, N_BOOT, seed)), "n_units": int(len(v)),
                        "frac_lower": float((v < 0).mean()), "frac_higher": float((v > 0).mean())}
            for c, ref in comps:
                for a, b in PAIRS:
                    da, db = unitdiff[(a, c, ref)], unitdiff[(b, c, ref)]
                    units = sorted(set(da) & set(db))
                    v = np.array([db[u] - da[u] for u in units])
                    V = np.full((len(rl), len(qs)), np.nan)
                    for u, x in zip(units, v):
                        V[rl.index(u[0]), qs.index(u[1])] = x
                    seed += 1
                    bs = rc.unit_boot_mean(v, N_BOOT, seed)
                    lo, hi = rc.ci(bs)
                    lo2, hi2 = rc.ci(rc.twoway_boot_mean(V, N_BOOT, seed))
                    blk["did"].setdefault(f"{c}-{ref}", {})[PKEY[(a, b)]] = {
                        "est": float(v.mean()), "ci": [lo, hi], "excludes_0": bool(lo > 0 or hi < 0),
                        "ci_twoway": [lo2, hi2], "excludes_0_twoway": bool(lo2 > 0 or hi2 < 0), "n_units": int(len(v))}
            G[sname] = blk
        out["groups"][g] = G
    return out


# ---------------------------------------------------------------------------
# 3. persona landscape
# ---------------------------------------------------------------------------

def landscape_obs(B: np.ndarray, d: np.ndarray) -> dict[str, np.ndarray]:
    c = B.mean(0)
    X, dd = B - c, d - c
    return rc.landscape_from_gram(X @ X.T, X @ dd, float(dd @ dd), rc.ident(len(B)))


def landscape_section(meta: dict, idx: np.ndarray, cats: dict) -> tuple[dict, dict]:
    disc = meta["disc_roles"]
    held = [r for r in heldout_roles(load_experiment())]
    out: dict = {}
    keep_for_figs: dict = {}
    for rep in REPS:
        for g in LGROUPS:
            sweep = {}
            for L in LAYERS:
                inp = {m: baseline_landscape_inputs(m, g, rep, L) for m in MODELS}
                roles50 = inp[MODELS[0]]["roles"]
                for m in MODELS:
                    if inp[m]["roles"] != roles50:
                        raise ValueError("role order differs across checkpoints")
                di = [roles50.index(r) for r in disc]
                row = {}
                for m in MODELS:
                    act = meta["act_norm"][m][rep][L]
                    o = landscape_obs(inp[m]["B_discovery"][di], inp[m]["d_discovery"])
                    row[SHORT[m]] = {"spread_over_act": float(o["spread"][0] / act), "pr": float(o["pr"][0]),
                                     "pc1_share": float(o["pc1_share"][0]),
                                     "abs_cos_pc1_assistant": float(o["abs_cos_pc1_assistant"][0]),
                                     "default_dist_over_spread": float(o["default_dist_over_spread"][0])}
                sweep[str(L)] = row
                if L != PRIMARY:
                    continue
                blk: dict = {"by_roleset": {}}
                for rs_name, ri in (("discovery40", di), ("all50", list(range(len(roles50))))):
                    n = len(ri)
                    bidx = idx if n == len(disc) else rc.boot_idx(n, N_BOOT, SEED + 7)
                    per, boot, jack = {}, {}, {}
                    jidx = rc.jack_idx(n)
                    for m in MODELS:
                        act = meta["act_norm"][m][rep][L]
                        B, d = inp[m]["B_discovery"][ri], inp[m]["d_discovery"]
                        c = B.mean(0)
                        X, dd = B - c, d - c
                        Gm, gd, d2 = X @ X.T, X @ dd, float(dd @ dd)
                        o = rc.landscape_from_gram(Gm, gd, d2, rc.ident(n))
                        bb = rc.landscape_from_gram(Gm, gd, d2, bidx)
                        jj = rc.landscape_from_gram(Gm, gd, d2, jidx)
                        stats = {"spread_over_act": (o["spread"][0] / act, bb["spread"] / act),
                                 "default_dist_over_spread": (o["default_dist_over_spread"][0], bb["default_dist_over_spread"]),
                                 "default_dist_over_act": (o["default_dist_over_spread"][0] * o["spread"][0] / act,
                                                           bb["default_dist_over_spread"] * bb["spread"] / act)}
                        jstats = {k: (o[k][0], jj[k]) for k in JACK_METS}
                        jack[m] = {k: v[1] for k, v in jstats.items()}
                        # noise-robust spread: discovery x validation cross-product (independent questions)
                        Bv = inp[m]["B_validation"][ri]
                        xs = float(np.mean(np.sum((B - B.mean(0)) * (Bv - Bv.mean(0)), 1)))
                        boot[m] = {k: v[1] for k, v in stats.items()}
                        dists = np.linalg.norm(B - d, axis=1)
                        order = np.argsort(dists)
                        roles_here = [roles50[i] for i in ri]
                        per[SHORT[m]] = {**{k: rc.summarise(*v) for k, v in stats.items()},
                                         **{k: rc.jack_summary(*v) for k, v in jstats.items()},
                                         "evr_top10": o["evr"][0][:10].tolist(),
                                         "spread_raw": float(o["spread"][0]), "act_norm": act,
                                         "spread_xsplit_over_act": float(np.sqrt(max(xs, 0)) / act),
                                         "nearest_to_default": [(roles_here[i], float(dists[i] / o["spread"][0]))
                                                                for i in order[:6]],
                                         "farthest_from_default": [(roles_here[i], float(dists[i] / o["spread"][0]))
                                                                   for i in order[-3:][::-1]],
                                         "default_dist_rank_pct": float((np.linalg.norm(B - c, axis=1)
                                                                         < np.linalg.norm(d - c)).mean())}
                    diffs = {PKEY[(a, b)]: {**{k: rc.diff_summary(per[SHORT[a]][k]["est"], per[SHORT[b]][k]["est"],
                                                                  boot[a][k], boot[b][k]) for k in boot[a]},
                                            **{k: rc.jack_diff(per[SHORT[a]][k]["est"], per[SHORT[b]][k]["est"],
                                                               jack[a][k], jack[b][k]) for k in jack[a]}}
                             for a, b in PAIRS}
                    blk["by_roleset"][rs_name] = {"per_checkpoint": per, "diffs": diffs}
                # sanity: our PCA vs the saved PCs / assistant contrast (discovery roles, discovery questions)
                san = {}
                for m in MODELS:
                    p = rc.pca(inp[m]["B_discovery"][di])
                    pcs = np.load(vec(m, f"{g}_{rep}_L{L}_baseline_pcs.npy")).astype(np.float64)
                    ac = np.load(vec(m, f"{g}_{rep}_L{L}_assistant_contrast.npy")).astype(np.float64)
                    a = inp[m]["d_discovery"] - inp[m]["B_discovery"][di].mean(0)
                    san[SHORT[m]] = {"abs_cos_pc1_vs_saved": float(abs(pcs[0] @ p["components"][0]) / np.linalg.norm(pcs[0])),
                                     "cos_assistant_vs_saved": float(a @ ac / np.linalg.norm(a) / np.linalg.norm(ac)),
                                     "saved_pca_evr1": meta["saved"][m][(rep, L, g, "persona_space")]["pca_evr"][0],
                                     "our_pca_evr1": float(p["evr"][0])}
                blk["sanity_vs_saved"] = san
                blk["cross_checkpoint"] = cross_checkpoint(inp, roles50, di, idx, cats)
                out.setdefault(rep, {})[g] = blk
                if g == "pooled":
                    keep_for_figs[rep] = {"inp": inp, "roles": roles50, "di": di}
            out[rep][g]["layer_sweep"] = sweep
        print(f"[landscape] {rep} done  (peak {peak_gb():.2f} GB)", flush=True)
    return out, keep_for_figs


def cross_checkpoint(inp: dict, roles50: list[str], di: list[int], idx: np.ndarray, cats: dict) -> dict:
    """Within-checkpoint geometry compared across checkpoints (no raw-coordinate alignment
    assumed), plus raw-coordinate sanity checks."""
    out: dict = {"rsa": {}, "pc1_scores": {}, "default_neighbourhood": {}, "reliability": {}, "raw_sanity": {}}
    Dm = {m: rc.dist(inp[m]["B_discovery"]) for m in MODELS}
    Dv = {m: rc.dist(inp[m]["B_validation"]) for m in MODELS}
    for m in MODELS:
        # within-checkpoint split-half reliability (discovery vs validation questions: a ceiling)
        out["reliability"][SHORT[m]] = {
            "rsa_disc_vs_val_all50": rc.spearman(rc.upper(Dm[m]), rc.upper(Dv[m])),
            "rsa_disc_vs_val_disc40": rc.spearman(rc.upper(Dm[m][np.ix_(di, di)]), rc.upper(Dv[m][np.ix_(di, di)]))}
    idx50 = rc.boot_idx(len(roles50), N_BOOT, SEED + 7)
    for a, b in PAIRS:
        k = PKEY[(a, b)]
        o50 = rc.spearman(rc.upper(Dm[a]), rc.upper(Dm[b]))
        o40 = rc.spearman(rc.upper(Dm[a][np.ix_(di, di)]), rc.upper(Dm[b][np.ix_(di, di)]))
        # across question sets as well (a's discovery questions vs b's validation questions): no shared question noise
        ox = 0.5 * (rc.spearman(rc.upper(Dm[a]), rc.upper(Dv[b])) + rc.spearman(rc.upper(Dv[a]), rc.upper(Dm[b])))
        rel = [out["reliability"][SHORT[a]]["rsa_disc_vs_val_all50"], out["reliability"][SHORT[b]]["rsa_disc_vs_val_all50"]]
        out["rsa"][k] = {"all50": rc.summarise(o50, rc.rsa_boot(Dm[a], Dm[b], idx50[:500])),
                         "all50_cross_split_disattenuated": float(ox / np.sqrt(rel[0] * rel[1])),
                         "disc40": rc.summarise(o40, rc.rsa_boot(Dm[a][np.ix_(di, di)], Dm[b][np.ix_(di, di)], idx[:500])),
                         "all50_cross_question_split": float(ox)}
        # PC1 role scores (fit on the 40 discovery roles in each checkpoint, all 50 projected)
        pa, pb = rc.pca(inp[a]["B_discovery"][di]), rc.pca(inp[b]["B_discovery"][di])
        sa = (inp[a]["B_discovery"] - pa["center"]) @ pa["components"][0]
        sb = (inp[b]["B_discovery"] - pb["center"]) @ pb["components"][0]
        Xa = inp[a]["B_discovery"][di] - inp[a]["B_discovery"][di].mean(0)
        Xb = inp[b]["B_discovery"][di] - inp[b]["B_discovery"][di].mean(0)
        jv = rc.pc1_scores_boot(Xa @ Xa.T, Xb @ Xb.T, rc.jack_idx(len(di)))
        out["pc1_scores"][k] = {"abs_corr_all50": float(abs(np.corrcoef(sa, sb)[0, 1])),
                                "abs_corr_disc40": rc.jack_summary(abs(np.corrcoef(sa[di], sb[di])[0, 1]), jv),
                                "eigengap_pc1_pc2": [float(pa["evr"][0] - pa["evr"][1]), float(pb["evr"][0] - pb["evr"][1])]}
        # who is near the default Assistant: Spearman of the 50 distances to the default
        da = np.linalg.norm(inp[a]["B_discovery"] - inp[a]["d_discovery"], axis=1)
        db = np.linalg.norm(inp[b]["B_discovery"] - inp[b]["d_discovery"], axis=1)
        out["default_neighbourhood"][k] = {"spearman_dist_to_default_all50": rc.spearman(da, db)}
        # raw-coordinate sanity (NOT used for any inference)
        Ba, Bb = inp[a]["B_discovery"], inp[b]["B_discovery"]
        Ca, Cb = Ba - Ba.mean(0), Bb - Bb.mean(0)
        cr = np.sum(Ba * Bb, 1) / np.linalg.norm(Ba, axis=1) / np.linalg.norm(Bb, axis=1)
        cc = np.sum(Ca * Cb, 1) / np.linalg.norm(Ca, axis=1) / np.linalg.norm(Cb, axis=1)
        aa = inp[a]["d_discovery"] - Ba[di].mean(0)
        ab = inp[b]["d_discovery"] - Bb[di].mean(0)
        sp = np.sqrt((Ca**2).sum(1).mean())
        out["raw_sanity"][k] = {"mean_cos_raw_centroids": float(cr.mean()),
                                "mean_cos_centred_centroids": float(cc.mean()),
                                "min_cos_centred_centroids": float(cc.min()),
                                "cos_mean_persona_vector": float(Ba.mean(0) @ Bb.mean(0) / np.linalg.norm(Ba.mean(0))
                                                                 / np.linalg.norm(Bb.mean(0))),
                                "cos_assistant_contrast": float(aa @ ab / np.linalg.norm(aa) / np.linalg.norm(ab)),
                                "rms_raw_centred_displacement_over_spread_a": float(
                                    np.sqrt(((Cb - Ca) ** 2).sum(1).mean()) / sp)}
    # which personas move most, dpo -> final (within-checkpoint geometry)
    a, b = MODELS[0], MODELS[-1]
    pa, pb = rc.pca(inp[a]["B_discovery"][di]), rc.pca(inp[b]["B_discovery"][di])
    sa = (inp[a]["B_discovery"] - pa["center"]) @ pa["components"][0]
    sb = (inp[b]["B_discovery"] - pb["center"]) @ pb["components"][0]
    if np.corrcoef(sa, sb)[0, 1] < 0:
        sb = -sb
    sa, sb = sa / sa[di].std(), sb / sb[di].std()
    rows = []
    spa = np.sqrt(((inp[a]["B_discovery"][di] - inp[a]["B_discovery"][di].mean(0)) ** 2).sum(1).mean())
    spb = np.sqrt(((inp[b]["B_discovery"][di] - inp[b]["B_discovery"][di].mean(0)) ** 2).sum(1).mean())
    ca, cb = inp[a]["B_discovery"][di].mean(0), inp[b]["B_discovery"][di].mean(0)
    for i, r in enumerate(roles50):
        oth = [j for j in range(len(roles50)) if j != i]
        prof = rc.spearman(Dm[a][i, oth], Dm[b][i, oth])
        rel = rc.spearman(Dm[a][i, oth], Dv[a][i, oth])
        dca = np.linalg.norm(inp[a]["B_discovery"][i] - ca) / spa
        dcb = np.linalg.norm(inp[b]["B_discovery"][i] - cb) / spb
        dda = np.linalg.norm(inp[a]["B_discovery"][i] - inp[a]["d_discovery"]) / spa
        ddb = np.linalg.norm(inp[b]["B_discovery"][i] - inp[b]["d_discovery"]) / spb
        rows.append({"role": r, "category": cats.get(r), "heldout": i not in di,
                     "profile_spearman_dpo_vs_final": prof, "profile_reliability_dpo_disc_vs_val": rel,
                     "dist_to_centre_over_spread": [float(dca), float(dcb)],
                     "dist_to_default_over_spread": [float(dda), float(ddb)],
                     "pc1_z": [float(sa[i]), float(sb[i])]})
    rows.sort(key=lambda x: x["profile_spearman_dpo_vs_final"])
    out["movers_dpo_to_final"] = rows
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6})


def save(fig, path: Path, title: str, note: str | None = None):
    fig.tight_layout(rect=(0, 0.06 if note else 0, 1, 0.93))
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=13, fontweight="bold")
    if note:
        fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=8.5, color=INK2, wrap=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def eb(est, lo, hi):
    """Error-bar half-widths; a percentile CI can exclude its point estimate (bootstrap bias), so clip at 0."""
    return [np.clip(est - lo, 0, None), np.clip(hi - est, 0, None)]


def fig_shifts(sh: dict, path: Path):
    keys = [("original", "auto_minus_baseline"), ("original", "auto_minus_neutral"),
            ("checkable", "auto_correct_minus_baseline"), ("checkable", "auto_correct_minus_neutral"),
            ("checkable", "evaluator_main")]
    fig, axes = plt.subplots(3, len(keys), figsize=(3.3 * len(keys), 8.4), sharex=True)
    x = np.arange(len(MODELS))
    for j, (g, nm) in enumerate(keys):
        for i, (met, ylab) in enumerate((("size", "|mean shift| / act. norm"), ("loo", "mean LOO cosine"),
                                         ("sep_xsplit", "separation ratio"))):
            ax = axes[i, j]
            for k, rep in enumerate(REPS):
                pc = sh[rep][str(PRIMARY)][g][nm]["per_checkpoint"]
                est = np.array([pc[m][met]["est"] for m in MODELS])
                lo = np.array([pc[m][met]["ci"][0] for m in MODELS])
                hi = np.array([pc[m][met]["ci"][1] for m in MODELS])
                xo = x + (k - 1) * 0.08
                ax.errorbar(xo, est, yerr=eb(est, lo, hi), color=RCOL[rep], marker="o", capsize=3,
                            label=REP_LABEL[rep] if (i == 0 and j == 0) else None)
            if met == "sep_xsplit":
                ax.axhline(1, color=INK2, lw=1, ls=":")
            if i == 0:
                ax.set_title(CLABEL[(g, nm)])
            if j == 0:
                ax.set_ylabel(ylab)
            ax.set_xticks(x, [MLABEL[m].replace(" (pre-RL)", "") for m in MODELS], fontsize=8.5)
    fig.legend(loc="upper right", ncol=3, bbox_to_anchor=(1, 0.965), fontsize=9)
    save(fig, path, f"Grading shifts across the RL trajectory (layer {PRIMARY})",
         "95% persona-bootstrap CIs (40 discovery personas, 2000 resamples). Separation ratio = persona-centroid separation under "
         "the condition / under the reference, from discovery × validation-question cross-products (question noise cancels); "
         "<1 = personas converge. Paired CIs on the differences in report.md.")


def fig_diffs(sh: dict, path: Path):
    rows = [(g, nm) for g in GROUPS for nm in CONTRASTS[g]]
    mets = (("size", "Δ |mean shift| / act. norm"), ("loo", "Δ mean LOO cosine"), ("sep_xsplit", "Δ separation ratio"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 7.6), sharey=True)
    y = np.arange(len(rows))[::-1]
    for i, (met, xl) in enumerate(mets):
        ax = axes[i]
        for k, rep in enumerate(REPS):
            for pk, mk, alpha in (("rl100-dpo", "s", 0.55), ("final-dpo", "o", 1.0)):
                est = np.array([sh[rep][str(PRIMARY)][g][nm]["diffs"][pk][met]["est"] for g, nm in rows])
                lo = np.array([sh[rep][str(PRIMARY)][g][nm]["diffs"][pk][met]["ci"][0] for g, nm in rows])
                hi = np.array([sh[rep][str(PRIMARY)][g][nm]["diffs"][pk][met]["ci"][1] for g, nm in rows])
                off = (k - 1) * 0.26 + (0.07 if pk == "final-dpo" else -0.07)
                ax.errorbar(est, y + off, xerr=eb(est, lo, hi), fmt=mk, color=RCOL[rep], alpha=alpha,
                            ms=4.5, capsize=2, lw=1.4,
                            label=f"{rep}: {pk.replace('-', ' − ')}" if i == 0 else None)
        ax.axvline(0, color=INK, lw=1)
        ax.set_xlabel(xl)
        for yy in y[:-1]:
            ax.axhline(yy - 0.5, color=GRID, lw=0.6)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [CLABEL[r] for r in rows])
    fig.legend(loc="upper right", ncol=3, bbox_to_anchor=(1, 0.965), fontsize=8.5)
    save(fig, path, f"Change with RL in each grading shift (layer {PRIMARY}): RL step 100 − DPO and RL final − DPO",
         "Paired persona bootstrap (same 40 discovery personas in every checkpoint, 2000 resamples), 95% CIs. "
         "Squares = step 100 − DPO (faded), circles = final − DPO.")


def fig_adherence(adh: dict, lens: dict, path: Path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10.6), gridspec_kw={"width_ratios": [3, 5]})
    for col, g in enumerate(GROUPS):
        blk = adh["groups"][g]["all"]
        keys = [k for k in blk["diffs"] if k.endswith("-baseline")]
        x = np.arange(len(keys))
        ax = axes[0, col]
        for k, m in enumerate(MODELS):
            d = [blk["diffs"][c][SHORT[m]] for c in keys]
            est = np.array([v["est"] for v in d])
            lo, hi = np.array([v["ci"][0] for v in d]), np.array([v["ci"][1] for v in d])
            ax.errorbar(x + (k - 1) * 0.18, est, yerr=eb(est, lo, hi), fmt="o", color=MCOL[m], capsize=3,
                        label=MLABEL[m] if col == 0 else None)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(x, [k.replace("-baseline", "") for k in keys], fontsize=8.5)
        ax.set_title(f"{'open-ended' if g == 'original' else 'checkable'}: condition − baseline (adherence, 0–3)")
        ax.set_ylabel("Δ role-adherence score")
        ax = axes[1, col]
        for k, (pk, colr) in enumerate((("rl100-dpo", MCOL[MODELS[1]]), ("final-dpo", MCOL[MODELS[2]]),
                                        ("final-rl100", "#e87ba4"))):
            d = [blk["did"][c][pk] for c in keys]
            est = np.array([v["est"] for v in d])
            lo, hi = np.array([v["ci"][0] for v in d]), np.array([v["ci"][1] for v in d])
            ax.errorbar(x + (k - 1) * 0.18, est, yerr=eb(est, lo, hi), fmt="D", color=colr, capsize=3, ms=5,
                        label=pk.replace("-", " − ") if col == 0 else None)
        ax.axhline(0, color=INK, lw=1)
        ax.set_xticks(x, [k.replace("-baseline", "") for k in keys], fontsize=8.5)
        ax.set_title("adherence: difference of differences across checkpoints")
        ax.set_ylabel("Δ(condition − baseline)")
        ax = axes[2, col]
        lb = lens["answer"]["groups"][g]["all"]
        for k, m in enumerate(MODELS):
            d = [lb["diffs"][c][SHORT[m]] for c in keys]
            est = np.exp([v["est"] for v in d])
            lo, hi = np.exp([v["ci"][0] for v in d]), np.exp([v["ci"][1] for v in d])
            ax.errorbar(x + (k - 1) * 0.18, est, yerr=eb(est, lo, hi), fmt="s", color=MCOL[m], capsize=3, ms=5)
        ax.axhline(1, color=INK, lw=1)
        ax.set_xticks(x, [k.replace("-baseline", "") for k in keys], fontsize=8.5)
        ax.set_title("answer length: geometric-mean ratio condition / baseline")
        ax.set_ylabel("answer tokens ratio")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper right", ncol=6, bbox_to_anchor=(1, 0.965), fontsize=8.5)
    save(fig, path, "Role adherence and answer length under grading context across the RL trajectory",
         "All 50 personas, prompt variant 0, one sample per (persona, question). 95% CIs: bootstrap over (persona, question) "
         "units, paired by unit within and across checkpoints. Two-way (persona × question) CIs, wider, are in report.md.")


def fig_landscape(ls: dict, path: Path, group: str = "pooled"):
    mets = (("spread_over_act", "rms spread / act. norm"), ("pr", "participation ratio"), ("pc1_share", "PC1 share"),
            ("abs_cos_pc1_assistant", "|cos(PC1, Assistant contrast)|"),
            ("default_dist_over_spread", "default Assistant dist. / spread"))
    fig, axes = plt.subplots(1, len(mets), figsize=(3.2 * len(mets), 3.9))
    x = np.arange(len(MODELS))
    for i, (met, yl) in enumerate(mets):
        ax = axes[i]
        for k, rep in enumerate(REPS):
            pc = ls[rep][group]["by_roleset"]["discovery40"]["per_checkpoint"]
            est = np.array([pc[SHORT[m]][met]["est"] for m in MODELS])
            lo = np.array([pc[SHORT[m]][met]["ci"][0] for m in MODELS])
            hi = np.array([pc[SHORT[m]][met]["ci"][1] for m in MODELS])
            ax.errorbar(x + (k - 1) * 0.08, est, yerr=eb(est, lo, hi), color=RCOL[rep], marker="o", capsize=3,
                        label=REP_LABEL[rep] if i == 0 else None)
        ax.set_title(yl)
        ax.set_xticks(x, ["DPO", "RL 100", "RL final"], fontsize=8.5)
    fig.legend(loc="upper right", ncol=3, bbox_to_anchor=(1, 0.985), fontsize=9)
    save(fig, path, f"Persona landscape (baseline) across the RL trajectory (layer {PRIMARY})",
         f"Baseline centroids of the 40 discovery personas, discovery questions, {group} over question groups. "
         "95% CIs over personas: bootstrap for spread and distances, leave-one-persona-out jackknife for participation ratio, PC1 share "
         "and PC1 alignment. Spread and PCA include question-sampling noise (noise-robust spread in report.md).")


def fig_maps(keep: dict, cats: dict, held: set, path: Path):
    fig, axes = plt.subplots(len(REPS), len(MODELS), figsize=(14, 4.3 * len(REPS)))
    for i, rep in enumerate(REPS):
        inp, roles, di = keep[rep]["inp"], keep[rep]["roles"], keep[rep]["di"]
        ref = None
        for j, m in enumerate(MODELS):
            ax = axes[i, j]
            B, d = inp[m]["B_discovery"], inp[m]["d_discovery"]
            p = rc.pca(B[di])
            S = (B - p["center"]) @ p["components"][:2].T
            sd = (d - p["center"]) @ p["components"][:2].T
            if ref is None:
                ref = S
            else:                                    # PC signs are arbitrary: align to DPO's role scores
                for k in range(2):
                    if np.corrcoef(S[:, k], ref[:, k])[0, 1] < 0:
                        S[:, k] *= -1
                        sd[k] *= -1
            for ci_, cat in enumerate(CATS):
                for hflag in (False, True):
                    sel = [n for n, r in enumerate(roles) if cats.get(r) == cat and (r in held) == hflag]
                    if not sel:
                        continue
                    ax.scatter(S[sel, 0], S[sel, 1], marker=CAT_MARK[cat], s=34,
                               facecolors="none" if hflag else SERIES[ci_], edgecolors=SERIES[ci_], lw=1.3,
                               label=(CAT_LABEL[cat] + (" (held out)" if hflag else "")) if (i == 0 and j == 0) else None)
            ax.scatter([sd[0]], [sd[1]], marker="*", s=260, color=INK, zorder=5,
                       label="default Assistant" if (i == 0 and j == 0) else None)
            ext = np.argsort(-np.hypot(S[:, 0] / S[:, 0].std(), S[:, 1] / S[:, 1].std()))[:7]
            for n in ext:
                ax.annotate(roles[n], (S[n, 0], S[n, 1]), fontsize=7.5, color=INK2, xytext=(3, 3),
                            textcoords="offset points")
            ax.set_title(f"{MLABEL[m]} · {rep}")
            ax.set_xlabel(f"PC1 ({100 * p['evr'][0]:.0f}%)")
            ax.set_ylabel(f"PC2 ({100 * p['evr'][1]:.0f}%)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=6, bbox_to_anchor=(1, 0.975), fontsize=8)
    save(fig, path, f"Persona maps: PC1/PC2 of baseline persona centroids (layer {PRIMARY})",
         "PCA fitted per checkpoint on the 40 discovery personas (filled); held-out personas (hollow) and the default Assistant "
         "(no system prompt, star) projected. Pooled over question groups, discovery questions. PC signs aligned to DPO.")


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

def fmt(v, nd=3):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def fd(d, nd=3):
    s = f"{d['est']:+.{nd}f} [{d['ci'][0]:+.{nd}f}, {d['ci'][1]:+.{nd}f}]"
    return f"**{s}**" if d.get("excludes_0") else s


def fe(d, nd=3):
    return f"{d['est']:.{nd}f} [{d['ci'][0]:.{nd}f}, {d['ci'][1]:.{nd}f}]"


def write_md(rep_: dict, path: Path, notes: list[str]):
    sh, adh, ls = rep_["shifts"], rep_["adherence"], rep_["landscape"]
    L = str(PRIMARY)
    o = ["# How does RLVR change the persona landscape and the effect of grading context? (OLMo-3 32B Think)", ""]
    o += ["Checkpoints of one trajectory: **DPO** (Olmo-3-32B-Think-DPO, before RLVR) → **RL step 100** "
          "(Olmo-3.1-32B-Think step_0100, ~4% of RLVR) → **RL final** (Olmo-3.1-32B-Think). "
          "Computed offline from saved per-persona displacements, condition centroids and judge scores "
          "(`scripts/olmo_rl_compare.py`). Bold = 95% CI excludes 0.", ""]
    o += ["## Summary", ""] + notes + [""]
    o += ["## Method notes", "",
          "* **Paired persona bootstrap**: one draw of the 40 discovery personas (2000 resamples, seed "
          f"{SEED}) is applied to all three checkpoints, so CIs on differences account for the shared persona sample. "
          "Questions are fixed (same 20 discovery questions per group in every checkpoint): CIs cover persona sampling only.",
          "* **Shift size** = ||mean_r D_r|| / that checkpoint's mean baseline activation norm (rep, layer); act. norm held "
          "fixed in resamples. **LOO** = mean over personas of cos(D_r, mean of the other personas' D). "
          "**Separation ratio** = mean pairwise squared distance between persona centroids under the contrast's "
          "positive state / under its negative state (plug-in, discovery questions; includes question-sampling "
          "noise, which pulls it towards 1). `sep_xsplit` replaces squared distances by discovery × validation "
          "cross-products (independent question sets, noise-free in expectation). The saved analysis' cross-fitted "
          "value (question-family split halves) is in report.json for comparison.",
          "* **Adherence**: judge score 0–3 of role adherence, all 50 personas, main wording grid, prompt variant 0, "
          "one sample per (persona, question). Differences are paired by (persona, question); CI = bootstrap over "
          "those units; `two-way` = personas and questions resampled independently (crossed design; wider).",
          "* **Landscape**: baseline centroids, discovery questions, pooled = mean of the two question groups' "
          "baselines. Cross-checkpoint comparisons use within-checkpoint geometry only (distance matrices, PC1 "
          "scores, distances to the default Assistant).", ""]

    def shift_table(met, title, nd=3, layer=L):
        t = [f"### {title} (layer {layer})", "",
             "| contrast | rep | DPO | RL 100 | RL final | RL100 − DPO | final − DPO | final − RL100 |",
             "|---|---|---|---|---|---|---|---|"]
        for g in GROUPS:
            for nm in CONTRASTS[g]:
                for rep in REPS:
                    b = sh[rep][layer][g][nm]
                    pc = b["per_checkpoint"]
                    t.append(f"| {CLABEL[(g, nm)]} | {rep} | " + " | ".join(fmt(pc[m][met]["est"], nd) for m in MODELS)
                             + " | " + " | ".join(fd(b["diffs"][k][met], nd) for k in ("rl100-dpo", "final-dpo", "final-rl100"))
                             + " |")
        return t + [""]

    o += ["## 1. Grading shifts", ""]
    o += shift_table("size", "Shift size |g| / activation norm", 3)
    o += shift_table("loo", "Mean leave-one-out cosine (how shared the shift is across personas)", 3)
    o += shift_table("sep", "Separation ratio (plug-in; < 1 = personas move closer together)", 3)
    o += shift_table("sep_xsplit", "Separation ratio, discovery × validation cross-product (noise-robust)", 3)
    o += [f"### Layer sweep: shift size, final − DPO (Δ |g|/act, 95% CI)", "",
          "| contrast | rep | " + " | ".join(f"L{x}" for x in LAYERS) + " |", "|---|---|" + "---|" * len(LAYERS)]
    for g, nm in [("original", "auto_minus_baseline"), ("original", "auto_minus_neutral"),
                  ("checkable", "auto_correct_minus_baseline"), ("checkable", "auto_correct_minus_neutral"),
                  ("checkable", "evaluator_main")]:
        for rep in REPS:
            o.append(f"| {CLABEL[(g, nm)]} | {rep} | " + " | ".join(
                fd(sh[rep][str(x)][g][nm]["diffs"]["final-dpo"]["size"], 3) for x in LAYERS) + " |")
    o += ["", f"### Layer sweep: mean LOO cosine, final − DPO", "",
          "| contrast | rep | " + " | ".join(f"L{x}" for x in LAYERS) + " |", "|---|---|" + "---|" * len(LAYERS)]
    for g, nm in [("original", "auto_minus_neutral"), ("checkable", "auto_correct_minus_neutral"),
                  ("checkable", "evaluator_main")]:
        for rep in REPS:
            o.append(f"| {CLABEL[(g, nm)]} | {rep} | " + " | ".join(
                fd(sh[rep][str(x)][g][nm]["diffs"]["final-dpo"]["loo"], 3) for x in LAYERS) + " |")
    o += [""]
    o += ["### Raw-basis sanity: cos(g_a, g_b) across checkpoints (layer 36)", "",
          "| contrast | rep | RL100 vs DPO | final vs DPO | final vs RL100 |", "|---|---|---|---|---|"]
    for g in GROUPS:
        for nm in CONTRASTS[g]:
            for rep in REPS:
                c = sh[rep][L][g][nm]["raw_cos_g_across_checkpoints"]
                o.append(f"| {CLABEL[(g, nm)]} | {rep} | " + " | ".join(fmt(c[k]) for k in ("rl100-dpo", "final-dpo", "final-rl100")) + " |")
    o += [""]

    o += ["## 2. Role adherence (judge score 0–3)", ""]
    for g in GROUPS:
        for sname in ("all", "discovery", "heldout"):
            blk = adh["groups"][g][sname]
            o += [f"### {g} questions, {sname} personas", "",
                  "| condition | mean DPO | mean RL100 | mean final | Δ vs ref: DPO | RL100 | final | "
                  "final − DPO (unit CI) | final − DPO (two-way CI) | RL100 − DPO | final − RL100 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for c in blk["means"]:
                if c == "baseline":
                    o.append(f"| baseline | " + " | ".join(fmt(blk["means"][c][SHORT[m]]["est"]) for m in MODELS)
                             + " | | | | | | | |")
            for k, v in blk["diffs"].items():
                c = k.split("-")[0]
                did = blk["did"][k]
                tw = did["final-dpo"]
                tws = f"{tw['est']:+.3f} [{tw['ci_twoway'][0]:+.3f}, {tw['ci_twoway'][1]:+.3f}]"
                if tw["excludes_0_twoway"]:
                    tws = f"**{tws}**"
                means = (" | ".join(fmt(blk["means"][c][SHORT[m]]["est"]) for m in MODELS)
                         if k.endswith("-baseline") else " | ".join("" for _ in MODELS))
                o.append(f"| {k} | {means} | " + " | ".join(fd({**v[SHORT[m]], "excludes_0": v[SHORT[m]]["ci"][0] > 0
                                                                or v[SHORT[m]]["ci"][1] < 0}) for m in MODELS)
                         + f" | {fd(did['final-dpo'])} | {tws} | {fd(did['rl100-dpo'])} | {fd(did['final-rl100'])} |")
            o += [""]

    ln = rep_["lengths"]
    o += ["## 2b. Answer and reasoning length (behavioural change that accompanies the shifts)", "",
          "Geometric-mean token ratio condition / reference (exp of the mean paired log ratio), all 50 personas; "
          "CI = bootstrap over (persona, question) units. Cross-checkpoint column = ratio of ratios final / DPO.", ""]
    for kind in ("answer", "reasoning"):
        o += [f"### {kind} tokens", "",
              "| group | comparison | DPO | RL 100 | RL final | final / DPO (ratio of ratios) | RL100 / DPO |",
              "|---|---|---|---|---|---|---|"]
        for g in GROUPS:
            blk = ln[kind]["groups"][g]["all"]
            for k, v in blk["diffs"].items():
                def ex(d):
                    s_ = f"{np.exp(d['est']):.2f} [{np.exp(d['ci'][0]):.2f}, {np.exp(d['ci'][1]):.2f}]"
                    return f"**{s_}**" if (d["ci"][0] > 0 or d["ci"][1] < 0) else s_
                o.append(f"| {g} | {k} | " + " | ".join(ex(v[SHORT[m]]) for m in MODELS)
                         + f" | {ex(blk['did'][k]['final-dpo'])} | {ex(blk['did'][k]['rl100-dpo'])} |")
        o += ["", "Raw mean tokens per condition: " + "; ".join(
            f"{k} " + "/".join(f"{v[SHORT[m]]:.0f}" for m in MODELS) for k, v in ln[kind]["raw_mean_tokens"].items())
            + " (DPO/RL100/final)", ""]
    o += ["### Does a persona's representation shift track its answer shortening? (layer 36)", "",
          "Pearson r across the 40 discovery personas between the persona's shift along the shared direction "
          "((D_r · ĝ) / act. norm) and its mean paired log answer-length ratio (negative r = personas whose answers "
          "shorten more shift more). 95% persona-bootstrap CI.", "",
          "| rep | contrast | DPO r [CI] | RL100 r [CI] | final r [CI] | mean log len ratio DPO / RL100 / final |",
          "|---|---|---|---|---|---|"]
    for rep in ("resp", "reason"):
        for k, v in rep_["length_link"][rep].items():
            o.append(f"| {rep} | {k} | " + " | ".join(fe(v[SHORT[m]], 2) for m in MODELS) + " | "
                     + " / ".join(f"{v[SHORT[m]]['mean_log_len_ratio']:+.2f}" for m in MODELS) + " |")
    o += [""]
    o += ["## 3. Persona landscape (baseline condition)", ""]
    mets = (("spread_over_act", "rms spread / act. norm"), ("pr", "participation ratio (jackknife CI)"),
            ("pc1_share", "PC1 share (jackknife CI)"),
            ("abs_cos_pc1_assistant", "abs cos(PC1, Assistant contrast) (jackknife CI)"),
            ("default_dist_over_spread", "default Assistant distance / spread"),
            ("default_dist_over_act", "default Assistant distance / act. norm"))
    for g in LGROUPS:
        for rs_name in ("discovery40", "all50"):
            o += [f"### {g}, {rs_name} (layer {L})", "",
                  "| metric | rep | DPO | RL 100 | RL final | RL100 − DPO | final − DPO | final − RL100 |",
                  "|---|---|---|---|---|---|---|---|"]
            for met, lab in mets:
                for rep in REPS:
                    b = ls[rep][g]["by_roleset"][rs_name]
                    o.append(f"| {lab} | {rep} | " + " | ".join(fe(b["per_checkpoint"][SHORT[m]][met]) for m in MODELS)
                             + " | " + " | ".join(fd(b["diffs"][k][met]) for k in ("rl100-dpo", "final-dpo", "final-rl100")) + " |")
            if rs_name == "discovery40":
                for rep in REPS:
                    b = ls[rep][g]["by_roleset"][rs_name]["per_checkpoint"]
                    o.append(f"| spread / act (disc × val, noise-robust) | {rep} | "
                             + " | ".join(fmt(b[SHORT[m]]["spread_xsplit_over_act"]) for m in MODELS) + " | | | |")
            o += [""]
    o += [f"### PCA spectrum (pooled, discovery40, layer {L}): explained-variance fractions of PCs 1–5", "",
          "| rep | checkpoint | PC1 | PC2 | PC3 | PC4 | PC5 |", "|---|---|---|---|---|---|---|"]
    for rep in REPS:
        for m in MODELS:
            ev = ls[rep]["pooled"]["by_roleset"]["discovery40"]["per_checkpoint"][SHORT[m]]["evr_top10"]
            o.append(f"| {rep} | {MLABEL[m]} | " + " | ".join(fmt(e) for e in ev[:5]) + " |")
    o += ["", f"### Nearest personas to the default Assistant (pooled, all 50, layer {L}; distance / spread)", "",
          "| rep | DPO | RL 100 | RL final |", "|---|---|---|---|"]
    for rep in REPS:
        pc = ls[rep]["pooled"]["by_roleset"]["all50"]["per_checkpoint"]
        o.append(f"| {rep} | " + " | ".join(", ".join(f"{r} {d:.2f}" for r, d in pc[SHORT[m]]["nearest_to_default"][:5])
                                         for m in MODELS) + " |")
    o += ["", f"### Cross-checkpoint similarity of the landscape (pooled, layer {L})", "",
          "| rep | pair | RSA all50 [CI] | RSA disc40 [CI] | RSA across question splits | same, disattenuated | "
          "abs corr PC1 scores disc40 [jackknife CI] | Spearman dist-to-default | raw cos centred centroids (sanity) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for rep in REPS:
        cc = ls[rep]["pooled"]["cross_checkpoint"]
        for k in ("rl100-dpo", "final-dpo", "final-rl100"):
            o.append(f"| {rep} | {k} | {fe(cc['rsa'][k]['all50'])} | {fe(cc['rsa'][k]['disc40'])} | "
                     f"{fmt(cc['rsa'][k]['all50_cross_question_split'])} | {fmt(cc['rsa'][k]['all50_cross_split_disattenuated'])} | "
                     f"{fe(cc['pc1_scores'][k]['abs_corr_disc40'])} | "
                     f"{fmt(cc['default_neighbourhood'][k]['spearman_dist_to_default_all50'])} | "
                     f"{fmt(cc['raw_sanity'][k]['mean_cos_centred_centroids'])} |")
    o += ["", "Within-checkpoint reliability ceiling (RSA of discovery-question vs validation-question centroids, all 50):", ""]
    for rep in REPS:
        cc = ls[rep]["pooled"]["cross_checkpoint"]["reliability"]
        o.append(f"* {rep}: " + ", ".join(f"{MLABEL[m]} {cc[SHORT[m]]['rsa_disc_vs_val_all50']:.3f}" for m in MODELS))
    o += ["", f"### Personas whose relative position changes most, DPO → final (pooled, layer {L})", "",
          "Lowest Spearman correlation of a persona's distance profile to the other 49 between DPO and final "
          "(within-checkpoint geometry), with its DPO split-half reliability for scale; distances in units of that "
          "checkpoint's spread.", "",
          "| rep | persona | profile ρ DPO vs final | DPO reliability | dist to centre DPO → final | dist to default DPO → final | PC1 z DPO → final |",
          "|---|---|---|---|---|---|---|"]
    for rep in REPS:
        for r in ls[rep]["pooled"]["cross_checkpoint"]["movers_dpo_to_final"][:6]:
            o.append(f"| {rep} | {r['role']}{' (held out)' if r['heldout'] else ''} | {r['profile_spearman_dpo_vs_final']:.2f} | "
                     f"{r['profile_reliability_dpo_disc_vs_val']:.2f} | {r['dist_to_centre_over_spread'][0]:.2f} → "
                     f"{r['dist_to_centre_over_spread'][1]:.2f} | {r['dist_to_default_over_spread'][0]:.2f} → "
                     f"{r['dist_to_default_over_spread'][1]:.2f} | {r['pc1_z'][0]:+.2f} → {r['pc1_z'][1]:+.2f} |")
    o += ["", "## 4. Reproduction of saved values (layer 36)", "",
          "| contrast | rep | checkpoint | size (ours / saved) | LOO (ours / saved) | sep plug-in / xsplit / saved cross-fit |",
          "|---|---|---|---|---|---|"]
    for g, nm in [("original", "auto_minus_neutral"), ("checkable", "auto_correct_minus_neutral")]:
        for rep in ("pre", "resp"):
            for m in MODELS:
                pc = sh[rep][L][g][nm]["per_checkpoint"][m]
                o.append(f"| {CLABEL[(g, nm)]} | {rep} | {MLABEL[m]} | {pc['size']['est']:.4f} / {pc['saved']['g_norm_over_act_norm']:.4f} | "
                         f"{pc['loo']['est']:.4f} / {pc['saved']['mean_loo_cos']:.4f} | {pc['sep']['est']:.3f} / "
                         f"{pc['sep_xsplit']['est']:.3f} / {fmt(pc['saved'].get('separation_ratio_crossfit'))} |")
    o += ["", "Saved adherence (summary.json, checkable auto_correct − baseline): " + ", ".join(
        f"{MLABEL[m]} {rep_['saved_adherence'][m]['checkable|auto_correct_minus_baseline']['mean']:+.3f}" for m in MODELS),
          "", "## Figures", ""]
    for f in sorted((path.parent / "figures").glob("*.png")):
        o.append(f"* `figures/{f.name}`")
    o += ["", f"Peak RSS of the analysis process: {rep_['peak_rss_gb']:.2f} GB.", ""]
    path.write_text("\n".join(o))


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(outputs_root() / "olmo_rl_compare"))
    ap.add_argument("--notes", default=None, help="optional markdown file with the plain-language summary")
    args = ap.parse_args()
    t0 = time.time()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    exp = load_experiment()
    if exp.get("design_name") != "olmo_rl_v1":
        raise SystemExit("set GSHIFT_EXPERIMENT=configs/experiment_olmo_rl.yaml")
    qmeta = {q["id"]: q for q in load_questions(exp)}
    cats = role_categories(exp)
    held = set(heldout_roles(exp))
    meta = load_meta(exp)
    print(f"[meta] loaded (peak {peak_gb():.2f} GB)", flush=True)
    idx = rc.boot_idx(len(meta["disc_roles"]), N_BOOT, SEED)
    report: dict = {"models": {m: MLABEL[m] for m in MODELS}, "pairs": [PKEY[p] for p in PAIRS],
                    "primary_layer": PRIMARY, "n_boot": N_BOOT, "seed": SEED,
                    "discovery_roles": meta["disc_roles"], "heldout_roles": sorted(held)}
    report["shifts"] = shifts_section(exp, meta, idx)
    report["adherence"], _ = adherence_section(exp, qmeta)
    print(f"[adherence] done (peak {peak_gb():.2f} GB)", flush=True)
    report["lengths"], lens = lengths_section(exp, qmeta)
    report["length_link"] = length_link(exp, meta, lens, idx)
    lens = None
    print(f"[lengths] done (peak {peak_gb():.2f} GB)", flush=True)
    report["landscape"], keep = landscape_section(meta, idx, cats)
    report["saved_adherence"] = {m: meta["saved_adherence"][m] for m in MODELS}
    report["peak_rss_gb"] = peak_gb()
    with open(out / "report.json", "w") as f:          # written before the figures (and again after)
        json.dump(r6(report), f, indent=1)
    style()
    fig_shifts(report["shifts"], out / "figures" / "a_shift_size_loo_sep.png")
    fig_diffs(report["shifts"], out / "figures" / "a2_shift_differences_forest.png")
    fig_adherence(report["adherence"], report["lengths"], out / "figures" / "b_adherence_and_length.png")
    fig_landscape(report["landscape"], out / "figures" / "c_landscape_structure.png")
    fig_maps(keep, cats, held, out / "figures" / "d_persona_maps.png")
    report["peak_rss_gb"] = peak_gb()
    report["runtime_s"] = time.time() - t0
    with open(out / "report.json", "w") as f:
        json.dump(r6(report), f, indent=1)
    notes = Path(args.notes).read_text().strip().splitlines() if args.notes else ["(summary pending)"]
    write_md(report, out / "report.md", notes)
    print(f"[done] {out} in {time.time() - t0:.0f}s, peak RSS {peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main()
