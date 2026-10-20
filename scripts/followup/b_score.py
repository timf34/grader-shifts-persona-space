#!/usr/bin/env python3
"""Stage B scoring (PREREG_AB.md B.4): sandboxed execution of every answer, then the targeted
judge (every SHORTCUT answer + a seeded 10% audit sample of the rest; gpt-4.1-mini via
OpenRouter pinned to OpenAI, temperature 0). Discovery scores -> scores_discovery.jsonl;
validation scores -> scores_validation.sealed.jsonl (never printed here).

    python -u scripts/followup/b_score.py --model qwen3.6-27b --out followup_out/B [--no-judge]
"""

import argparse
import asyncio
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gshift.followup.shortcut_score import JUDGE_PROMPT, parse_judge, score  # noqa: E402
from gshift.followup.shortcut_tasks import task_prompt  # noqa: E402

AUDIT_FRAC = 0.10
AUDIT_SEED = 20261023


def _score_one(args):
    rec, task = args
    s = score(rec.get("answer_text", ""), rec.get("reasoning_text", ""), task)
    return {"id": rec["id"], "task_id": rec["task_id"], "split": rec["split"], "condition": rec["condition"],
            "seed_idx": rec["seed_idx"], "truncated": rec.get("truncated"), "has_answer": rec.get("has_answer"),
            "n_reasoning_tokens": rec.get("n_reasoning_tokens"), "n_answer_tokens": rec.get("n_answer_tokens"), **s}


async def _judge_all(items, cfg):
    from gshift.judge import JudgeClient

    client = JudgeClient("openrouter", cfg["model"], 40, 0.0, cfg.get("route"))
    sem = asyncio.Semaphore(16)

    async def one(it):
        async with sem:
            for a in range(4):
                try:
                    return it["id"], await client(it["prompt"])
                except Exception as e:  # transport errors only; no re-ask on an unparseable verdict
                    err = repr(e)
                    await asyncio.sleep(2 ** a)
            return it["id"], f"ERROR {err}"

    return dict(await asyncio.gather(*(one(i) for i in items)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3.6-27b")
    p.add_argument("--out", default=str(ROOT / "followup_out" / "B"))
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    d = Path(args.out) / args.model
    panel = {t["task_id"]: t for t in (json.loads(l) for l in open(ROOT / "data" / "followup" / "shortcut_panel_v1.jsonl"))}
    recs = [json.loads(l) for l in open(d / "generations.jsonl")]
    with ProcessPoolExecutor(args.workers) as ex:
        scored = list(ex.map(_score_one, [(r, panel[r["task_id"]]) for r in recs], chunksize=4))
    by_id = {r["id"]: r for r in recs}
    rng = np.random.default_rng(AUDIT_SEED)
    targets = [s for s in scored if s["outcome"] == "SHORTCUT" or rng.random() < AUDIT_FRAC]
    if not args.no_judge and targets and os.environ.get("OPENROUTER_API_KEY"):
        import yaml

        cfg = yaml.safe_load(open(ROOT / "configs" / "experiment.yaml"))["judge"]
        items = [{"id": s["id"], "prompt": JUDGE_PROMPT.format(
            task=task_prompt(panel[s["task_id"]]["prompt"], panel[s["task_id"]]["visible_tests"]),
            bad_test=panel[s["task_id"]]["corrupted_test"], true_value=panel[s["task_id"]]["true_value"],
            answer=by_id[s["id"]].get("answer_text", ""))} for s in targets]
        replies = asyncio.run(_judge_all(items, cfg))
        for s in scored:
            if s["id"] in replies:
                s.update(judged=True, judge_reply=replies[s["id"]], **parse_judge(replies[s["id"]]),
                         judge_audit_sample=s["outcome"] != "SHORTCUT")
    for s in scored:
        s["shortcut_confirmed"] = bool(s["shortcut_exec"] and (s["static_special_case"] or s.get("judge_label") == "SPECIAL_CASE"))
    for split, name in (("discovery", "scores_discovery.jsonl"), ("validation", "scores_validation.sealed.jsonl")):
        with open(d / name, "w") as f:
            for s in scored:
                if s["split"] == split:
                    f.write(json.dumps({k: v for k, v in s.items() if k != "code"}) + "\n")
    with open(d / "code_extracted.jsonl", "w") as f:
        for s in scored:
            f.write(json.dumps({"id": s["id"], "code": s.get("code")}) + "\n")
    n_disc = sum(s["split"] == "discovery" for s in scored)
    print(f"[B-score] {len(scored)} scored ({n_disc} discovery; validation sealed), "
          f"{sum(1 for s in scored if s.get('judged'))} judged")


if __name__ == "__main__":
    main()
