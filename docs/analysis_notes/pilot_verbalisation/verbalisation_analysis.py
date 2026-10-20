"""Pilot: do thinking traces verbalise the grading cue, and does that track displacement size?

Exploratory, read-only analysis of the thinking on/off pilot (Qwen3.6-27B, Qwen3.8-27B;
10 pilot roles + default). No GPU, no API calls.

Usage (from the repo root):
    .venv/bin/python docs/analysis_notes/pilot_verbalisation/verbalisation_analysis.py \
        [--data-dir DIR]   # local mirror of the HF dataset (contains pilot/); fetched if absent
        [--layer 36] [--seed 0]

Outputs go to docs/analysis_notes/pilot_verbalisation/out/ (CSV tables, hand-check samples,
and summary.md with every table the README quotes).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = HERE / "out"
DATASET = "timf34/grader-shifts-persona-space-outputs"
MODELS = ["qwen3.6-27b", "qwen3.8-27b"]
GROUPS = {
    "original": ["auto", "human"],
    "checkable": ["auto_correct", "human_correct", "auto_quality", "human_quality"],
}
GRADED = [c for cs in GROUPS.values() for c in cs]
FOCUS = [("original", "auto"), ("checkable", "auto_correct"), ("checkable", "human_quality")]
PILOT_ROLES = ["analyst", "caregiver", "comedian", "cyborg", "evaluator", "hacker",
               "mentor", "musician", "robot", "strategist"]

# --------------------------------------------------------------------------------------
# Regexes (case-insensitive). BROAD is the brief's suggested set (with a leading \b so that
# e.g. "upgrade" does not count). CUE is a narrower set meant to catch references to *the
# response being evaluated/graded* rather than generic uses of "evaluate"/"assess"/"score".
# --------------------------------------------------------------------------------------
BROAD = (r"\b(evaluat|grad(e|er|ed|ing)|assess|automat(ed|ic)|checker|reviewer|"
         r"human (reader|grader|evaluator)|reference answer|scor(e|ed|ing))")
_V = r"(evaluat\w*|grad\w*|assess\w*|check\w*|scor\w*|judg\w*|review\w*|rat(e|er|ers|ing)|eval)"
CUE_PATTERNS = {
    # "will be evaluated", "is going to be graded", "to be assessed automatically" ...
    "passive_future": r"\b(will|would|is going to|gonna|to) be (automatically |manually )?"
                      r"(evaluated|graded|assessed|checked|rated|scored|judged|reviewed|marked)\b",
    "being_graded": r"\bbeing (evaluated|graded|assessed|scored|judged|rated)\b",
    # "automated system", "automatic grader", "automated evaluation", "automatic eval"
    "automated_x": r"\bautomat(ed|ic)(ally)?[- ](" + _V[1:-1] + r"|systems?|evaluators?|tools?|metrics?|pipelines?)\b",
    # "evaluated automatically", "graded by a human", "assessed by an automated system"
    "by_agent": r"\b(evaluated|graded|assessed|checked|scored|rated|judged|reviewed) (automatically|by "
                r"(a |an |the )?(human|person|people|automat\w*|machine|system|grader|evaluator|reviewer|assessor))",
    "grader": r"\bgraders?\b",
    "graded_grading": r"\b(graded|grading)\b",
    "auto_eval": r"\bauto[- ]?eval\w*",
    "x_evaluated": r"\b(human|system|auto|machine|person)-(evaluated|assessed|graded|rated|checked|scored)\b",
    "rate_quality": r"\brat(e|es|ed|ing) (the |my |its )?(overall )?quality\b",
    # "human evaluator", "human grader", "human reader", "person grading"
    "human_agent": r"\b(human|person)[- ](evaluator|grader|reviewer|reader|assessor|judge|rater|evaluation|"
                   r"grading|assessment|review)s?\b",
    # "a person will assess", "the assessor will evaluate", "someone will check"
    "agent_will": r"\b(person|human|user|reader|assessor|evaluator|grader|reviewer|someone)s? will \"?" + _V,
    # "the system will assess my answer", "evaluators expect type hints", "a good evaluator would look for"
    "system_will": r"\b(system|evaluators?|graders?) (will|would|might|may|expects?|prefers?|wants?|likely) \"?"
                   r"(evaluat\w*|grad\w*|assess\w*|check\w*|scor\w*|judg\w*|review\w*|rat(e|ing)|expect\w*|prefer\w*|"
                   r"look\w*|parse\w*|want\w*)",
    "reference_answer": r"\breference answer",
    # meta-talk about the cue: "the evaluation note", "grading instruction", "assessment prefix"
    "cue_meta": r"\b(evaluation|grading|assessment|grader|evaluator)[- ](note|instruction|prefix|statement|line|"
                r"context|constraint|requirement|aspect|mention|cue|clause|condition|setup|framing)s?\b",
    # "Evaluation: based on overall quality", "the evaluation is based on ..."
    "eval_based_on": r"\b(evaluation|grading|assessment)\**\s*(:\**\s*|is |will be )based on\b",
}
BROAD_RX = re.compile(BROAD, re.I)
CUE_RX = {k: re.compile(v, re.I) for k, v in CUE_PATTERNS.items()}
CUE_ANY = re.compile("|".join(f"(?:{v})" for v in CUE_PATTERNS.values()), re.I)


def cue_hits(text: str) -> list[str]:
    return [k for k, rx in CUE_RX.items() if rx.search(text or "")]


# --------------------------------------------------------------------------------------
def fetch(data_dir: Path | None) -> Path:
    if data_dir is not None and (data_dir / "pilot").exists():
        return data_dir
    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download
    load_dotenv(REPO / ".env")
    pats = ["pilot/*/*/*/responses/*", "pilot/*/*/*/activations/*/meta.json",
            "pilot/*/*/*/activations/*/index.jsonl", "pilot/thinking_on/*/*/activations/*/resp.npy",
            "pilot/thinking_on/*/*/activations/*/reason.npy", "pilot/thinking_off/*/*/activations/*/resp.npy"]
    p = snapshot_download(DATASET, repo_type="dataset", allow_patterns=pats,
                          local_dir=str(data_dir) if data_dir else None, token=os.environ.get("HF_TOKEN"))
    return Path(p)


def load_responses(root: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((root / "pilot").glob("thinking_*/*/*/responses/*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            rows.append({k: r.get(k) for k in (
                "id", "role", "condition", "question_group", "question_id", "prompt_variant", "wording_id",
                "prefix", "baseline_id", "enable_thinking", "model_key", "reasoning_text", "answer_text",
                "has_answer", "n_reasoning_tokens", "n_answer_tokens", "truncated")})
    df = pd.DataFrame(rows)
    df["thinking"] = np.where(df["enable_thinking"], "on", "off")
    for c in ("reasoning_text", "answer_text"):
        df[c] = df[c].fillna("")
    df["broad"] = df["reasoning_text"].map(lambda t: bool(BROAD_RX.search(t)))
    df["cue_hits"] = df["reasoning_text"].map(cue_hits)
    df["cue"] = df["cue_hits"].map(bool)
    df["cue_answer"] = df["answer_text"].map(lambda t: bool(CUE_ANY.search(t)))
    a = df["answer_text"]
    df["answer_chars"] = a.str.len()
    df["n_headers"] = a.str.count(r"(?m)^\s*#{1,6} ")
    df["n_bullets"] = a.str.count(r"(?m)^\s*([-*+]|\d+[.)]) ")
    df["n_bold"] = a.str.count(r"\*\*") // 2
    df["is_graded"] = df["condition"] != "baseline"
    return df


def snippet(text: str, rx: re.Pattern, width: int = 60) -> str:
    m = rx.search(text)
    if not m:
        return ""
    s = text[max(0, m.start() - width): m.end() + width].replace("\n", " ")
    return re.sub(r"\s+", " ", s)


def md(df: pd.DataFrame, floatfmt: str = ".2f") -> str:
    """Minimal markdown table (no tabulate dependency)."""
    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(v, floatfmt)
        if isinstance(v, tuple):
            return " / ".join(map(str, v))
        return str(v)
    idx_name = " / ".join(str(n) for n in df.index.names if n is not None) or ""
    head = [idx_name] + [fmt(c) for c in df.columns]
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for i, row in df.iterrows():
        out.append("| " + " | ".join([fmt(i)] + [fmt(v) for v in row.tolist()]) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------------------
def part1_rates(df: pd.DataFrame, rep: list[str]) -> pd.DataFrame:
    on = df[df.thinking == "on"]
    t = (on.groupby(["model_key", "question_group", "condition"])
           .agg(n=("id", "size"), broad=("broad", "mean"), cue=("cue", "mean"),
                cue_in_answer=("cue_answer", "mean"), med_reason_tok=("n_reasoning_tokens", "median"))
           .reset_index())
    order = {c: i for i, c in enumerate(["baseline"] + GRADED)}
    t = t.sort_values(["model_key", "question_group", "condition"], key=lambda s: s.map(order) if s.name == "condition" else s)
    off = df[df.thinking == "off"].groupby(["model_key", "question_group", "condition"])["cue_answer"].mean()
    t["cue_in_answer_thinking_off"] = [off.get((m, g, c), np.nan) for m, g, c in
                                       zip(t.model_key, t.question_group, t.condition)]
    t.to_csv(OUT / "rates_by_model_condition.csv", index=False)
    rep.append("## Part 1a. Verbalisation rate by model x condition (thinking on)\n")
    rep.append("`broad` / `cue` = fraction of reasoning traces matching the regex; `cue_in_answer` = CUE regex on the "
               "final answer text (thinking on), `cue_in_answer_thinking_off` = same on thinking-off answers.\n")
    rep.append(md(t.set_index(["model_key", "question_group", "condition"])))
    # per-pattern firing rates
    rows = []
    for (m, cond), sub in on.groupby(["model_key", "condition"]):
        r = {"model_key": m, "condition": cond, "n": len(sub)}
        for k in CUE_PATTERNS:
            r[k] = float(sub["cue_hits"].map(lambda h: k in h).mean())
        rows.append(r)
    pp = pd.DataFrame(rows).set_index(["model_key", "condition"])
    pp.to_csv(OUT / "cue_pattern_rates.csv")
    rep.append("\n### Which CUE sub-patterns fire (fraction of traces)\n")
    rep.append(md(pp))
    # length-matched check: verbalisation by reasoning-length tercile (graded only)
    g = on[on.is_graded].copy()
    labs = ["Q1 short", "Q2", "Q3", "Q4 long"]
    g["len_bin"] = g.groupby("model_key")["n_reasoning_tokens"].transform(
        lambda x: pd.qcut(x, 4, labels=labs).astype(str))
    lb = g.pivot_table(index="model_key", columns="len_bin", values="cue", aggfunc="mean")[labs]
    rng = g.groupby(["model_key", "len_bin"])["n_reasoning_tokens"].agg(["min", "max"])
    rep.append("\n### CUE rate in graded traces by reasoning-length quartile (quartiles within model)\n")
    rep.append(md(lb) + "\n\nQuartile token ranges: " +
               "; ".join(f"{m} {b}: {int(r['min'])}-{int(r['max'])}" for (m, b), r in rng.iterrows()))
    return t


def part1_roles(df: pd.DataFrame, rep: list[str]) -> pd.DataFrame:
    on = df[df.thinking == "on"]
    rows = []
    for (m, role), sub in on.groupby(["model_key", "role"]):
        r = {"model_key": m, "role": role}
        for grp, conds in GROUPS.items():
            b = sub[(sub.condition == "baseline") & (sub.question_group == grp)]
            r[f"{grp}:baseline"] = b["cue"].mean()
            for c in conds:
                r[c] = sub[sub.condition == c]["cue"].mean()
        r["broad:baseline(all)"] = sub[sub.condition == "baseline"]["broad"].mean()
        r["broad:graded(all)"] = sub[sub.is_graded]["broad"].mean()
        rows.append(r)
    t = pd.DataFrame(rows).set_index(["model_key", "role"])
    t.to_csv(OUT / "cue_rate_by_role.csv")
    rep.append("\n## Part 1b. CUE verbalisation rate by role (thinking on)\n")
    rep.append("Each graded cell has n=30 traces (default: n=10). Baseline columns are the role's false-positive floor.\n")
    rep.append(md(t))
    return t


def part1_handcheck(df: pd.DataFrame, rep: list[str], seed: int) -> None:
    rnd = random.Random(seed)
    on = df[df.thinking == "on"]
    g = on[on.is_graded]
    hits = g[g.cue].to_dict("records")
    non = {m: g[(~g.cue) & (g.model_key == m)].to_dict("records") for m in MODELS}
    bhits = on[(on.condition == "baseline") & on.cue].to_dict("records")
    broad_only = g[g.broad & ~g.cue].to_dict("records")
    lines = ["# Hand-check samples (seed %d)\n" % seed]
    lines.append("## 15 random CUE hits in graded conditions\n")
    for r in rnd.sample(hits, 15):
        lines.append(f"- [{r['model_key']} {r['condition']} {r['role']} {r['id']}] patterns={r['cue_hits']}\n  "
                     f"prefix: {r['prefix']}\n  ...{snippet(r['reasoning_text'], CUE_ANY)}...\n")
    lines.append("\n## All CUE hits in BASELINE traces (false-positive floor)\n")
    for r in bhits:
        lines.append(f"- [{r['model_key']} {r['role']} {r['question_id']}] patterns={r['cue_hits']}\n  "
                     f"...{snippet(r['reasoning_text'], CUE_ANY)}...\n")
    lines.append("\n## 12 random BROAD-but-not-CUE hits in graded conditions\n")
    for r in rnd.sample(broad_only, min(12, len(broad_only))):
        lines.append(f"- [{r['model_key']} {r['condition']} {r['role']}]\n  ...{snippet(r['reasoning_text'], BROAD_RX)}...\n")
    lines.append("\n## 10 random graded NON-hits (5 per model), full reasoning text\n")
    for m in MODELS:
        for r in rnd.sample(non[m], 5):
            lines.append(f"### [{m} {r['condition']} {r['role']} {r['id']}] prefix: {r['prefix']}\n")
            lines.append("```\n" + r["reasoning_text"].strip() + "\n```\n")
    (OUT / "handcheck_samples.md").write_text("\n".join(lines))
    rep.append(f"\nHand-check samples written to out/handcheck_samples.md "
               f"({len(hits)} graded CUE hits, {len(bhits)} baseline CUE hits, {len(broad_only)} graded broad-only hits).")


# --------------------------------------------------------------------------------------
def within(y, lab, codes):
    """Verbalised-minus-non mean of y within strata, harmonic-n weighted over strata with both kinds."""
    k = codes.max() + 1
    n1 = np.bincount(codes, weights=lab, minlength=k)
    n0 = np.bincount(codes, weights=~lab, minlength=k)
    s1 = np.bincount(codes, weights=y * lab, minlength=k)
    s0 = np.bincount(codes, weights=y * ~lab, minlength=k)
    ok = (n1 > 0) & (n0 > 0)
    if not ok.any():
        return np.nan, 0
    w = n1[ok] * n0[ok] / (n1[ok] + n0[ok])
    return float(np.average(s1[ok] / n1[ok] - s0[ok] / n0[ok], weights=w)), int(ok.sum())

def perm_test(y, lab, codes, n_perm, rng):
    obs, nst = within(y, lab, codes)
    if not np.isfinite(obs):
        return obs, np.nan, nst
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, bounds)
    null = np.empty(n_perm)
    for b in range(n_perm):
        pl = lab.copy()
        for gi in groups:
            pl[gi] = lab[rng.permutation(gi)]
        null[b] = within(y, pl, codes)[0]
    return obs, float((np.abs(null) >= abs(obs) - 1e-12).mean()), nst



# --------------------------------------------------------------------------------------
def part2_answers(df: pd.DataFrame, rep: list[str], seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    on = df[(df.thinking == "on")].copy()
    base = on[on.condition == "baseline"].set_index(["model_key", "id"])
    g = on[on.is_graded & on.has_answer.astype(bool)].copy()
    feats = ["n_answer_tokens", "answer_chars", "n_headers", "n_bullets", "n_bold", "n_reasoning_tokens"]
    for f in feats:
        keys = pd.MultiIndex.from_arrays([g["model_key"], g["baseline_id"]])
        g["d_" + f] = g[f].to_numpy() - base[f].reindex(keys).to_numpy()
    rows = []
    for (m, grp), sub in g.groupby(["model_key", "question_group"]):
        v, nv = sub[sub.cue], sub[~sub.cue]
        r = {"model_key": m, "group": grp, "n_verb": len(v), "n_nonverb": len(nv)}
        for f in ["n_answer_tokens", "answer_chars", "n_headers", "n_bullets", "n_bold", "n_reasoning_tokens"]:
            r[f"{f}: verb"] = v[f].mean()
            r[f"{f}: non"] = nv[f].mean()
        for f in ["n_answer_tokens", "n_reasoning_tokens"]:
            r[f"d_{f} vs base: verb"] = v["d_" + f].mean()
            r[f"d_{f} vs base: non"] = nv["d_" + f].mean()
        # within-stratum difference: strata = condition x role x question (3 prompt variants each)
        codes = pd.factorize(sub.condition + "|" + sub.role + "|" + sub.question_id)[0]
        lab = sub.cue.to_numpy(bool)
        for f in ["d_n_answer_tokens", "n_headers", "n_bullets", "d_n_reasoning_tokens"]:
            obs, pv, nst = perm_test(sub[f].to_numpy(float), lab, codes, 2000, rng)
            r[f"within(cond,role,q) diff {f}"] = obs
            r[f"within(cond,role,q) perm p {f}"] = pv
        r["n_strata"] = nst
        r["MWU p (answer tok, unstratified)"] = (stats.mannwhitneyu(v["n_answer_tokens"], nv["n_answer_tokens"]).pvalue
                                                 if len(v) > 2 and len(nv) > 2 else np.nan)
        rows.append(r)
    t = pd.DataFrame(rows).set_index(["model_key", "group"])
    t.to_csv(OUT / "answer_features_verb_vs_non.csv")
    rep.append("\n## Part 2. Answer features, verbalised (CUE) vs non-verbalised graded traces (thinking on)\n")
    rep.append("`d_x vs base` = change relative to the matched baseline unit (same role, question, prompt variant). "
               "`within(cond,role,q)` = verbalised minus non-verbalised mean inside condition x role x question strata "
               "(3 prompt variants each) that contain both kinds, harmonic-n weighted; `perm p` = label permutation "
               "within strata (2000 perms).\n")
    rep.append(md(t.T, ".1f"))
    # reference: condition-level answer length, thinking on vs off
    h = df[df.has_answer.astype(bool)].copy()
    h["cell"] = np.where(h.condition == "baseline", "baseline_" + h.question_group, h.condition)
    ref = h.groupby(["model_key", "thinking", "cell"])["n_answer_tokens"].mean().unstack("cell")
    ref = ref[["baseline_original", "auto", "human", "baseline_checkable"] + GROUPS["checkable"]]
    rep.append("\n### Reference: mean answer tokens by condition (thinking on vs off)\n")
    rep.append(md(ref, ".0f"))
    return t


# --------------------------------------------------------------------------------------
class Acts:
    def __init__(self, root: Path, layer: int):
        self.root, self.layer, self.cache = root, layer, {}

    def get(self, thinking: str, model: str, cond: str, role: str, rep: str) -> dict[str, np.ndarray]:
        key = (thinking, model, cond, role, rep)
        if key not in self.cache:
            d = self.root / "pilot" / f"thinking_{thinking}" / model / cond / "activations" / role
            meta = json.loads((d / "meta.json").read_text())
            li = meta["layers"].index(self.layer)
            arr = np.load(d / f"{rep}.npy", mmap_mode="r")
            out = {}
            for line in (d / "index.jsonl").open():
                r = json.loads(line)
                ok = (r["n_reasoning_tokens"] or 0) > 0 if rep == "reason" else \
                     (bool(r["has_answer"]) and (r["n_answer_tokens"] or 0) > 0)
                if ok:
                    out[r["id"]] = np.asarray(arr[r["row"], li, :], dtype=np.float64)
            self.cache[key] = out
        return self.cache[key]


def unit_disps(acts: Acts, df: pd.DataFrame, thinking: str, model: str, cond: str, role: str, rep: str):
    """Per-unit displacement graded - matched baseline. Returns (ids, D[U,H], cue flags, baseline norms)."""
    g = acts.get(thinking, model, cond, role, rep)
    b = acts.get(thinking, model, "baseline", role, rep)
    sub = df[(df.thinking == thinking) & (df.model_key == model) & (df.condition == cond) & (df.role == role)]
    ids, D, flags, bn = [], [], [], []
    for r in sub.itertuples():
        if r.id in g and r.baseline_id in b:
            ids.append(r.id)
            D.append(g[r.id] - b[r.baseline_id])
            flags.append(bool(r.cue))
            bn.append(np.linalg.norm(b[r.baseline_id]))
    return ids, np.array(D), np.array(flags), np.array(bn)


def part3_displacement(df: pd.DataFrame, root: Path, layer: int, rep_md: list[str], seed: int) -> None:
    acts = Acts(root, layer)
    have_off = any((root / "pilot" / "thinking_off").glob("*/*/activations/*/resp.npy"))
    roles_all = PILOT_ROLES + ["default"]
    role_rows, unit_rows, corr_rows = [], [], []
    qid_of = dict(zip(df["id"], df["question_id"]))
    vec_store, goff_store = {}, {}
    for rep in ["resp", "reason"]:
        for m in MODELS:
            for grp, conds in GROUPS.items():
                for cond in conds:
                    per = {}
                    for role in roles_all:
                        ids, D, fl, bn = unit_disps(acts, df, "on", m, cond, role, rep)
                        per[role] = (ids, D, fl, bn)
                    dr = {r: per[r][1].mean(0) for r in roles_all}
                    gsum = sum(dr[r] for r in PILOT_ROLES)
                    g_all = gsum / len(PILOT_ROLES)
                    off_dr = {}
                    if rep == "resp" and have_off:
                        for role in roles_all:
                            _, Doff, _, _ = unit_disps(acts, df, "off", m, cond, role, "resp")
                            off_dr[role] = Doff.mean(0)
                        g_off = sum(off_dr[r] for r in PILOT_ROLES) / len(PILOT_ROLES)
                        goff_store[(m, cond)] = g_off
                    if rep == "resp":
                        vec_store[(m, cond)] = (np.concatenate([per[r][1] for r in PILOT_ROLES]),
                                                np.concatenate([per[r][2] for r in PILOT_ROLES]),
                                                np.concatenate([[r] * len(per[r][0]) for r in PILOT_ROLES]))
                    for role in roles_all:
                        ids, D, fl, bn = per[role]
                        n = len(D)
                        d = dr[role]
                        resid = ((D - d) ** 2).sum() / (n * (n - 1))
                        corr_sq = float(d @ d - resid)
                        g_loo = (gsum - d) / (len(PILOT_ROLES) - 1) if role in PILOT_ROLES else g_all
                        gh = g_loo / np.linalg.norm(g_loo)
                        proj = D @ gh
                        row = {"rep": rep, "model_key": m, "group": grp, "contrast": f"{cond}_minus_baseline",
                               "role": role, "n_units": n, "cue_rate": float(fl.mean()),
                               "base_cue_rate": float(df[(df.thinking == "on") & (df.model_key == m) & (df.role == role) &
                                                         (df.condition == "baseline") & (df.question_group == grp)]["cue"].mean()),
                               "disp_norm": float(np.linalg.norm(d)),
                               "disp_norm_rel": float(np.linalg.norm(d) / bn.mean()),
                               "disp_norm_noisecorr": float(np.sign(corr_sq) * np.sqrt(abs(corr_sq))),
                               "proj_on_g_loo": float(proj.mean()),
                               "cos_d_g_loo": float(d @ gh / np.linalg.norm(d))}
                        if off_dr:
                            o = off_dr[role]
                            row["cos_on_off_role"] = float(d @ o / np.linalg.norm(d) / np.linalg.norm(o))
                            row["cos_g_on_off"] = float(g_all @ g_off / np.linalg.norm(g_all) / np.linalg.norm(g_off))
                        role_rows.append(row)
                        for i, f, p, Du in zip(ids, fl, proj, D):
                            unit_rows.append({"rep": rep, "model_key": m, "group": grp, "condition": cond, "role": role,
                                              "id": i, "question_id": qid_of[i], "cue": f, "proj": float(p),
                                              "unit_norm": float(np.linalg.norm(Du))})
    R = pd.DataFrame(role_rows)
    U = pd.DataFrame(unit_rows)
    R.to_csv(OUT / "displacement_by_role.csv", index=False)
    U.to_csv(OUT / "unit_projections.csv", index=False)

    # ---- across-role Spearman (10 pilot roles) ----
    for (rep, m, grp, con), sub in R[R.role.isin(PILOT_ROLES)].groupby(["rep", "model_key", "group", "contrast"]):
        r = {"rep": rep, "model_key": m, "contrast": con, "cue_rate_range": f"{sub.cue_rate.min():.2f}-{sub.cue_rate.max():.2f}"}
        for y in ["disp_norm", "disp_norm_noisecorr", "disp_norm_rel", "proj_on_g_loo", "cos_d_g_loo", "cos_on_off_role"]:
            if y in sub and sub[y].notna().all() and sub.cue_rate.nunique() > 1:
                rho, p = stats.spearmanr(sub.cue_rate, sub[y])
                r[f"rho({y})"] = rho
                r[f"p({y})"] = p
        excess = sub.cue_rate - sub.base_cue_rate
        if excess.nunique() > 1:
            r["rho(excess_rate, disp_norm)"] = stats.spearmanr(excess, sub.disp_norm)[0]
        corr_rows.append(r)
    C = pd.DataFrame(corr_rows).set_index(["rep", "model_key", "contrast"])
    C.to_csv(OUT / "spearman_rate_vs_displacement.csv")

    rep_md.append(f"\n## Part 3. Verbalisation rate vs displacement (layer {layer}, thinking on)\n")
    show = R[(R.rep == "resp")]
    focus = [f"{c}_minus_baseline" for _, c in FOCUS]
    cols = ["cue_rate", "base_cue_rate", "disp_norm", "disp_norm_noisecorr", "disp_norm_rel", "cos_d_g_loo"]
    if "cos_on_off_role" in show:
        cols.append("cos_on_off_role")
    for m in MODELS:
        for con in focus:
            s = show[(show.model_key == m) & (show.contrast == con)].set_index("role")[cols]
            extra = ""
            if "cos_g_on_off" in show:
                extra = f" (cos(g_on, g_off) over 10 roles = {show[(show.model_key == m) & (show.contrast == con)].cos_g_on_off.iloc[0]:.2f})"
            rep_md.append(f"\n### {m} | rep resp | {con}{extra}\n")
            rep_md.append(md(s, ".3f"))
    rep_md.append("\n### Spearman across the 10 pilot roles: CUE rate vs displacement measures (exploratory, n=10)\n")
    rep_md.append(md(C, ".2f"))

    # ---- per-unit split: verbalised vs not, projection onto leave-one-role-out g ----
    rows = []
    rng = np.random.default_rng(seed)
    for (rep, m, grp, cond), sub in U.groupby(["rep", "model_key", "group", "condition"]):
        sub = sub[sub.role.isin(PILOT_ROLES)]
        lab = sub.cue.to_numpy(bool)
        c_role = pd.factorize(sub.role)[0]
        c_rq = pd.factorize(sub.role + "|" + sub.question_id)[0]
        r = {"rep": rep, "model_key": m, "contrast": f"{cond}_minus_baseline",
             "n_verb": int(lab.sum()), "n_non": int((~lab).sum())}
        for y in ["proj", "unit_norm"]:
            yy = sub[y].to_numpy(float)
            r[f"{y}: verb"] = yy[lab].mean() if lab.any() else np.nan
            r[f"{y}: non"] = yy[~lab].mean() if (~lab).any() else np.nan
            for nm, codes in (("role", c_role), ("role x q", c_rq)):
                obs, p, nst = perm_test(yy, lab, codes, 2000, rng)
                r[f"{y}: diff within {nm}"] = obs
                r[f"{y}: p within {nm}"] = p
                if y == "proj":
                    r[f"n strata ({nm})"] = nst
        rows.append(r)
    S = pd.DataFrame(rows).set_index(["rep", "model_key", "contrast"])
    S.to_csv(OUT / "unit_split_verb_vs_non.csv")
    rep_md.append("\n### Per-unit split: projection of (graded - baseline) onto leave-one-role-out g, verbalised vs not\n")
    rep_md.append("`diff within role` / `within role x q` = verbalised minus non-verbalised, averaged within role "
                  "(or within role x question; strata = 3 prompt variants) with harmonic-n weights over strata containing "
                  "both kinds; `p` = two-sided label-permutation p within those strata (2000 perms). 10 pilot roles only.\n")
    rep_md.append(md(S.T, ".2f"))

    # ---- direction test: do verbalised units' mean displacements align less with the thinking-OFF direction? ----
    if goff_store:
        rows = []
        for (m, cond), (D, fl, rl) in vec_store.items():
            go = goff_store[(m, cond)]
            go = go / np.linalg.norm(go)
            n1, n0 = int(fl.sum()), int((~fl).sum())
            r = {"model_key": m, "contrast": f"{cond}_minus_baseline", "n_verb": n1, "n_non": n0}
            k = min(n1, n0)
            if k >= 5:
                cv, cn = [], []
                i1, i0 = np.flatnonzero(fl), np.flatnonzero(~fl)
                for _ in range(500):
                    a = D[rng.choice(i1, k, replace=False)].mean(0)
                    b = D[rng.choice(i0, k, replace=False)].mean(0)
                    cv.append(a @ go / np.linalg.norm(a)); cn.append(b @ go / np.linalg.norm(b))
                r["subsample n"] = k
                r["cos(mean D_verb, g_off)"] = float(np.mean(cv))
                r["cos(mean D_non, g_off)"] = float(np.mean(cn))
                r["diff (verb-non)"] = float(np.mean(np.array(cv) - np.array(cn)))
                r["frac subsamples verb<non"] = float(np.mean(np.array(cv) < np.array(cn)))
            rows.append(r)
        T = pd.DataFrame(rows).set_index(["model_key", "contrast"])
        T.to_csv(OUT / "verb_vs_non_alignment_with_thinking_off.csv")
        rep_md.append("\n### Direction test: alignment of verbalised vs non-verbalised units with the thinking-OFF direction\n")
        rep_md.append("Mean (graded - baseline) over equal-size random subsamples (k = min group size, 500 draws, pooled over "
                      "the 10 pilot roles), cosine with g_off (thinking-off mean displacement over the same 10 roles, all units).\n")
        rep_md.append(md(T, ".2f"))


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--layer", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-acts", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    root = fetch(a.data_dir)
    df = load_responses(root)
    rep = ["# Pilot verbalisation analysis: generated tables\n",
           f"BROAD regex: `{BROAD}`\n", "CUE regex (union of):\n"]
    rep += [f"- `{k}`: `{v}`" for k, v in CUE_PATTERNS.items()]
    rep.append("")
    part1_rates(df, rep)
    part1_roles(df, rep)
    part1_handcheck(df, rep, a.seed)
    part2_answers(df, rep, a.seed)
    if not a.skip_acts:
        part3_displacement(df, root, a.layer, rep, a.seed)
    (OUT / "summary.md").write_text("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
