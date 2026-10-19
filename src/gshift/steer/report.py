"""Behavioural summary per model: every metric reported separately (legitimate performance,
exploitation, refusal, length, formatting), as paired differences from the unsteered
condition with bootstrap CIs over prompts, and with the spread of norm-matched random
directions as a reference band."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io import read_jsonl, write_json
from .metrics import response_metrics
from .tasks import gsm8k_correct

PRIMARY = {
    "sorh": ["gap", "metric", "quality", "no_answer", "refusal", "n_answer_tokens", "n_reasoning_tokens",
             "n_bullets", "n_headers", "n_bold"],
    "sorh_descored": ["quality", "no_answer", "refusal", "n_answer_tokens", "n_reasoning_tokens", "n_bullets",
                      "n_headers", "n_bold"],
    "gsm8k": ["correct", "no_answer", "n_answer_tokens", "n_reasoning_tokens", "truncated"],
}


def per_sample_table(gen_root: Path, judge_path: Path | None) -> pd.DataFrame:
    rows = []
    judged = {}
    if judge_path is not None and judge_path.exists():
        for r in read_jsonl(judge_path):
            judged[(r["id"], r["condition"])] = r
    for bdir in sorted(p for p in gen_root.iterdir() if p.is_dir()):
        for f in sorted(bdir.glob("*.jsonl")):
            for r in read_jsonl(f):
                m = response_metrics(r)
                row = {"id": r["id"], "benchmark": r["benchmark"], "condition": r["condition"],
                       "direction": r.get("direction") or r["condition"], "signed_strength": r.get("signed_strength", 0.0),
                       **m, "no_answer": float(not r.get("has_answer"))}
                if r["benchmark"] == "gsm8k":
                    c = gsm8k_correct(r.get("answer_text", ""), r["meta"]["answer"]) if r.get("has_answer") else None
                    row["correct"] = float(bool(c))          # missing answer counts as wrong
                    row["correct_answered"] = float(bool(c)) if r.get("has_answer") else np.nan
                if r["benchmark"].startswith("sorh"):
                    j = judged.get((r["id"], r["condition"]), {})
                    for k in ("metric", "quality", "gap"):
                        row[k] = j.get(k) if j.get(k) is not None else np.nan
                    row["unparsed"] = float(bool(j.get("unparsed")))
                rows.append(row)
    return pd.DataFrame(rows)


def paired_effects(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for bench, sub in df.groupby("benchmark"):
        metrics = [m for m in PRIMARY.get(bench, []) if m in sub]
        base = sub[sub["condition"] == "none"].set_index("id")
        for cond, s in sub.groupby("condition"):
            s = s.set_index("id")
            for m in metrics:
                common = s.index.intersection(base.index)
                d = (s.loc[common, m] - base.loc[common, m]).dropna().to_numpy()
                val = s[m].dropna().to_numpy()
                if not len(val):
                    continue
                rec = {"benchmark": bench, "condition": cond, "direction": s["direction"].iloc[0],
                       "signed_strength": float(s["signed_strength"].iloc[0]), "metric": m,
                       "mean": float(val.mean()), "n": int(len(val))}
                if len(d) and cond != "none":
                    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
                    rec.update(diff=float(d.mean()), diff_lo=float(np.quantile(bs, 0.025)),
                               diff_hi=float(np.quantile(bs, 0.975)), n_paired=int(len(d)))
                out.append(rec)
    return pd.DataFrame(out)


def random_band(eff: pd.DataFrame) -> pd.DataFrame:
    """Per (benchmark, metric, |strength|): mean/sd of random-direction effects, and each
    recovered direction's effect as a z-score against that band."""
    r = eff[eff["direction"].str.startswith("random", na=False) & eff["diff"].notna()].copy()
    r["abs_strength"] = r["signed_strength"].abs()
    band = r.groupby(["benchmark", "metric", "abs_strength"])["diff"].agg(["mean", "std", "count"]).reset_index()
    o = eff[~eff["direction"].str.startswith("random", na=False) & eff["diff"].notna()].copy()
    o["abs_strength"] = o["signed_strength"].abs()
    o = o.merge(band, on=["benchmark", "metric", "abs_strength"], how="left", suffixes=("", "_random"))
    o["z_vs_random"] = (o["diff"] - o["mean_random"]) / o["std"]
    return o


def behaviour_summary(model_key: str, steer_dir: Path) -> dict[str, Any]:
    df = per_sample_table(steer_dir / "gen", steer_dir / "sorh_judgements.jsonl")
    df = df[df["benchmark"] != "persona"]
    (steer_dir / "tables").mkdir(exist_ok=True)
    df.to_csv(steer_dir / "tables" / "per_sample.csv", index=False)
    eff = paired_effects(df)
    eff.to_csv(steer_dir / "tables" / "effects_vs_unsteered.csv", index=False)
    vs = random_band(eff)
    vs.to_csv(steer_dir / "tables" / "effects_vs_random_band.csv", index=False)
    summ = {"model_key": model_key, "n_samples": int(len(df)),
            "conditions": sorted(df["condition"].unique().tolist()),
            "effects": eff.to_dict(orient="records"), "vs_random": vs.to_dict(orient="records"),
            "notes": ["Natural-context effects are rows natural_auto / natural_human (unsteered + Q1 prefix).",
                      "GSM8K 'correct' counts a missing/truncated answer as wrong; 'no_answer' reports it separately.",
                      "SoRH metric/quality/gap exclude no-answer and unparsed samples (reported separately)."]}
    write_json(steer_dir / "behaviour_summary.json", summ)
    _plot(eff, steer_dir / "figures")
    return summ


def _plot(eff: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    for (bench, metric), sub in eff[eff["diff"].notna()].groupby(["benchmark", "metric"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        for d, s in sub.groupby("direction"):
            s = s.sort_values("signed_strength")
            if str(d).startswith("random"):
                ax.plot(s["signed_strength"], s["diff"], "o", color="grey", alpha=0.5)
            elif str(d).startswith("natural"):
                continue
            else:
                ax.errorbar(s["signed_strength"], s["diff"], yerr=[s["diff"] - s["diff_lo"], s["diff_hi"] - s["diff"]],
                            fmt="o-", label=d, capsize=3)
        for c, mk in (("natural_auto", "^"), ("natural_human", "v")):
            row = sub[sub["condition"] == c]
            if len(row):
                ax.axhline(row["diff"].iloc[0], ls=":", lw=1, label=c)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("signed strength (x mean residual norm)")
        ax.set_ylabel(f"{metric}: steered - unsteered")
        ax.set_title(f"{bench}: {metric} (grey = random directions)", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{bench}_{metric}.png", dpi=130)
        plt.close(fig)
