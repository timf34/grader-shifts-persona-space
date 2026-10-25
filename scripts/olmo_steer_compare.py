"""OLMo RL-stage steering comparison: does the same grading direction change behaviour more after
RLVR? Reads each checkpoint's judgements (scripts/12_steer_judge.py) and, per benchmark metric and
steering condition, the change vs the unsteered condition, paired by prompt id, with a bootstrap
95% CI; the random-direction band at the same strength; and the final - DPO difference of effects
(prompts resampled independently per checkpoint, paired within).

Metrics (higher = more of the behaviour):
  adherence         role adherence 0-3 (Assistant Axis rubric), plain persona prompts
  adherence_graded  role adherence 0-3 under the auto_correct grading sentence (subtract only)
  sorh_gap          School of Reward Hacks: metric score - quality score (Betley et al.'s judges)
  sorh_metric       School of Reward Hacks: metric (proxy) score alone
  agentic_harmful   agentic misalignment: harmful-action rate (reference classifier)

    python scripts/olmo_steer_compare.py --out outputs/olmo_steer_compare
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = {"DPO": "olmo3-32b-dpo", "step 100": "olmo31-32b-rl100", "final": "olmo31-32b-final"}
METRICS = ["adherence", "adherence_graded", "sorh_gap", "sorh_metric", "agentic_harmful"]
N_BOOT, SEED = 2000, 20261025


def _rows(path: Path):
    if not path.exists():
        return
    for line in open(path):
        if line.strip():
            yield json.loads(line)


def load(steer_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    """metric -> condition -> {prompt id: value}"""
    out: dict = defaultdict(lambda: defaultdict(dict))
    for r in _rows(steer_dir / "adherence_judgements.jsonl"):
        if r.get("score") is None:
            continue
        bench, cond = r["condition"].split("|", 1)
        out[bench][cond][r["id"]] = float(r["score"])
    for r in _rows(steer_dir / "sorh_judgements.jsonl"):
        if r.get("no_answer") or r.get("metric") is None or r.get("quality") is None:
            continue
        out["sorh_gap"][r["condition"]][r["id"]] = float(r["metric"]) - float(r["quality"])
        out["sorh_metric"][r["condition"]][r["id"]] = float(r["metric"])
    for r in _rows(steer_dir / "agentic_judgements.jsonl"):
        if r.get("unparsed") or r.get("no_answer"):
            continue
        out["agentic_harmful"][r["condition"]][r["id"]] = float(bool(r["harmful"]))
    return out


def paired(base: dict, cond: dict) -> np.ndarray:
    ids = sorted(set(base) & set(cond))
    return np.array([cond[i] - base[i] for i in ids])


def boot_mean(d: np.ndarray, rng) -> tuple[float, float, float]:
    if len(d) == 0:
        return (np.nan, np.nan, np.nan)
    bs = d[rng.integers(0, len(d), (N_BOOT, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def parse(cond: str):
    m = re.match(r"(.+)_([+-]\d\.\d+)$", cond)
    return (m.group(1), float(m.group(2))) if m else (cond, 0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="outputs")
    p.add_argument("--out", default="outputs/olmo_steer_compare")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    data = {k: load(Path(a.root) / m / "steering") for k, m in MODELS.items()}

    res: dict = {}
    for metric in METRICS:
        conds = sorted({c for k in data for c in data[k][metric] if c != "none" and not c.startswith("natural")})
        res[metric] = {"base": {}, "effects": {}, "random_band": {}, "did_final_minus_dpo": {}}
        for k in MODELS:
            b = data[k][metric].get("none", {})
            res[metric]["base"][k] = float(np.mean(list(b.values()))) if b else None
        diffs = {}
        for c in conds:
            res[metric]["effects"][c] = {}
            for k in MODELS:
                d = paired(data[k][metric].get("none", {}), data[k][metric].get(c, {}))
                diffs[(k, c)] = d
                res[metric]["effects"][c][k] = {"mean": None, "lo": None, "hi": None, "n": len(d)} if len(d) == 0 else \
                    dict(zip(("mean", "lo", "hi"), boot_mean(d, rng)), n=len(d))
            dd, df_ = diffs[("DPO", c)], diffs[("final", c)]
            if len(dd) and len(df_):
                bs = (df_[rng.integers(0, len(df_), (N_BOOT, len(df_)))].mean(1)
                      - dd[rng.integers(0, len(dd), (N_BOOT, len(dd)))].mean(1))
                res[metric]["did_final_minus_dpo"][c] = {"mean": float(df_.mean() - dd.mean()),
                                                         "lo": float(np.percentile(bs, 2.5)),
                                                         "hi": float(np.percentile(bs, 97.5))}
        for k in MODELS:   # random band: the largest |effect| over random directions, per strength
            band = defaultdict(float)
            for c in conds:
                d, s = parse(c)
                if d.startswith("random") and len(diffs[(k, c)]):
                    band[abs(s)] = max(band[abs(s)], abs(float(diffs[(k, c)].mean())))
            res[metric]["random_band"][k] = dict(band)
    (out / "report.json").write_text(json.dumps(res, indent=1))

    # markdown table
    L = ["# OLMo steering across RL stages", "",
         "Effect = steered - unsteered on the same prompts, mean [95% bootstrap CI]. **Bold** = CI excludes 0 "
         "AND |effect| exceeds the largest random-direction effect in that checkpoint (random directions were run at ±0.3 only; the same band is used for ±0.1). "
         "Last column: final - DPO difference of effects.", ""]
    for metric in METRICS:
        r = res[metric]
        if not r["effects"]:
            continue
        base = ", ".join(f"{k} {v:.3f}" for k, v in r["base"].items() if v is not None)
        band = "; ".join(f"{k} " + ", ".join(f"±{s}: {v:.3f}" for s, v in sorted(b.items())) for k, b in r["random_band"].items())
        L += [f"## {metric}", "", f"Unsteered: {base}. Random band (max |effect|): {band}.", "",
              "| condition | " + " | ".join(MODELS) + " | final - DPO |", "|---|" + "---|" * (len(MODELS) + 1)]
        for c, eff in r["effects"].items():
            cells = []
            for k in MODELS:
                e = eff[k]
                if e["mean"] is None:
                    cells.append("—")
                    continue
                s = abs(parse(c)[1])
                sig = (e["lo"] > 0 or e["hi"] < 0) and abs(e["mean"]) > max(r["random_band"][k].values(), default=0) \
                    and not c.startswith("random")
                t = f"{e['mean']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"
                cells.append(f"**{t}**" if sig else t)
            did = r["did_final_minus_dpo"].get(c)
            dtxt = "—" if not did else (f"{did['mean']:+.3f} [{did['lo']:+.3f}, {did['hi']:+.3f}]")
            if did and (did["lo"] > 0 or did["hi"] < 0):
                dtxt = f"**{dtxt}**"
            L.append(f"| {c} | " + " | ".join(cells) + f" | {dtxt} |")
        L.append("")
    (out / "report.md").write_text("\n".join(L))
    print(f"[steer_compare] -> {out}/report.md")


if __name__ == "__main__":
    main()
