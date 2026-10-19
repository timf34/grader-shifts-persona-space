#!/usr/bin/env python3
"""Summarise a smoke run: token budgets, truncation, reasoning/answer splitting, samples.
Use it to set generation.max_tokens in configs/experiment.yaml before the full run.

    python scripts/smoke_report.py --model qwen3.6-27b            # reads outputs_smoke/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--root", default=os.environ.get("GSHIFT_SMOKE_OUTPUTS", str(ROOT / "outputs_smoke")))
    args = p.parse_args()
    os.environ["GSHIFT_OUTPUTS"] = args.root
    from gshift.io import conditions_present, read_jsonl, responses_path, roles_present

    rows = []
    for c in conditions_present(args.model):
        for r in roles_present(args.model, c):
            rows += read_jsonl(responses_path(args.model, c, r))
    if not rows:
        sys.exit(f"no smoke generations under {args.root}/{args.model}")
    gen = np.array([len(r["gen_ids"]) for r in rows])
    reas = np.array([r["n_reasoning_tokens"] for r in rows])
    ans = np.array([r["n_answer_tokens"] for r in rows])
    budget = max(r["max_tokens"] for r in rows)
    print(f"{len(rows)} smoke generations, budget {budget} tokens")
    for name, a in (("generated", gen), ("reasoning", reas), ("answer", ans)):
        print(f"  {name:9s} tokens: p50={np.percentile(a, 50):.0f} p90={np.percentile(a, 90):.0f} "
              f"p99={np.percentile(a, 99):.0f} max={a.max()}")
    print(f"  truncated: {np.mean([r['truncated'] for r in rows]):.1%}   "
          f"has_answer: {np.mean([r['has_answer'] for r in rows]):.1%}   "
          f"think_closed: {np.mean([r['think_closed'] for r in rows]):.1%}")
    for g in sorted({r["question_group"] for r in rows}):
        a = np.array([len(r["gen_ids"]) for r in rows if r["question_group"] == g])
        print(f"  {g:10s} generated p90={np.percentile(a, 90):.0f} max={a.max()}")
    rec = int(np.ceil(np.percentile(gen, 99) * 1.5 / 512) * 512)
    print(f"suggested generation.max_tokens >= {rec} (1.5 x p99 of the smoke run; note the smoke budget "
          f"itself was {budget}, so truncated smoke rows understate the tail)")
    print("\nsample (first record):")
    r = rows[0]
    print("PROMPT:", r["prompt_text"][-300:])
    print("REASONING:", r["reasoning_text"][:300].replace("\n", " "), "...")
    print("ANSWER:", r["answer_text"][:500])


if __name__ == "__main__":
    main()
