"""Role adherence under steering (benchmarks `adherence` and `adherence_graded`): the Q1 judge
(configs/experiment.yaml `judge`: gpt-4.1-mini, Assistant Axis rubric, 0-3), on the visible
final answer, with the ORIGINAL question (no grading sentence), exactly as in Q1.

Judgements: steering/adherence_judgements.jsonl, one row per (benchmark|condition, id); the
`condition` field is "<benchmark>|<condition>" because the two benchmarks share prompt ids.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import numpy as np

from ..io import append_jsonl, read_jsonl
from ..judge import _score_all, build_judge_prompt, make_client


def judge_adherence(gen_files: list[Path], out_path: Path, judge_cfg: dict[str, Any]) -> int:
    done = {(r["condition"], r["id"]) for r in read_jsonl(out_path) if r.get("score") is not None or r.get("skipped")}
    items, skipped = [], []
    for f in gen_files:
        for r in read_jsonl(f):
            key = f"{r['benchmark']}|{r['condition']}"
            if (key, r["id"]) in done:
                continue
            base = {"id": r["id"], "role": r["meta"]["role"], "condition": key, "question_id": r["meta"]["question_id"]}
            if not r.get("has_answer") or not (r.get("answer_text") or "").strip():
                skipped.append({**base, "score": None, "skipped": "no_answer"})
                continue
            items.append({**base, "out": str(out_path),
                          "prompt": build_judge_prompt(r["meta"]["role"], r["meta"]["question"], r["answer_text"])})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if skipped:
        append_jsonl(out_path, skipped)
    print(f"[adherence-judge] {len(items)} to judge, {len(skipped)} without an answer", flush=True)
    if items:
        asyncio.run(_score_all(make_client(judge_cfg), items, int(judge_cfg.get("concurrency", 32))))
    return len(items)


def adherence_summary(path: Path, n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Per benchmark x condition x question group: mean score, share fully in character (3),
    paired difference from the benchmark's `none` condition with a bootstrap CI over prompts.
    No-answer rows are reported separately and excluded from the means."""
    latest: dict[tuple[str, str], dict] = {}
    for r in read_jsonl(path):
        latest[(r["condition"], r["id"])] = r
    by: dict[str, dict[str, dict]] = {}
    for (key, i), r in latest.items():
        by.setdefault(key, {})[i] = r
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for key, rows in sorted(by.items()):
        bench, cond = key.split("|", 1)
        ref = by.get(f"{bench}|none", {})
        res: dict[str, Any] = {}
        for g in ("original", "checkable", "all"):
            ids = [i for i in rows if g == "all" or str(rows[i]["question_id"]).startswith("chk_") == (g == "checkable")]
            sc = {i: rows[i]["score"] for i in ids if rows[i].get("score") is not None}
            if not sc:
                continue
            v = np.array(list(sc.values()), float)
            e = {"n": len(v), "n_no_answer": sum(1 for i in ids if rows[i].get("skipped")),
                 "mean": float(v.mean()), "frac_fully": float((v == 3).mean())}
            paired = [(sc[i], ref[i]["score"]) for i in sc if i in ref and ref[i].get("score") is not None]
            if cond != "none" and len(paired) >= 5:
                d = np.array([a - b for a, b in paired], float)
                bs = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(1)
                e.update({"diff_vs_none": float(d.mean()), "diff_ci95": np.percentile(bs, [2.5, 97.5]).tolist(),
                          "n_paired": len(d)})
            res[g] = e
        out.setdefault(bench, {})[cond] = res
    return out
