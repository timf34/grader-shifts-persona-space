#!/usr/bin/env python3
"""Role-adherence judging via OpenRouter's Batch API (gshift.judge_batch): same judge as
scripts/03_judge.py at ~half the price, main-grid records only. Resumable at every step.

    python -u scripts/judge_batch.py --model qwen3.6-27b submit      # build + POST batches
    python -u scripts/judge_batch.py --model qwen3.6-27b poll        # write finished results
    python -u scripts/judge_batch.py --model qwen3.6-27b wait        # poll every 5 min until done,
                                                                     # resubmitting unparseable verdicts
    python -u scripts/judge_batch.py --model qwen3.6-27b submit --limit 5   # tiny live test

Needs the model's responses locally (scripts/download_outputs.py --model M --what all, or
run it on the pod). Writes outputs/<model>/<condition>/scores/<role>.jsonl.
"""

import json
import time

from _common import base_parser, experiment_for

from gshift.io import conditions_present, roles_present
from gshift.judge_batch import collect_items, load_state, poll, resubmit_unparsed, submit


def main():
    p = base_parser(__doc__)
    p.add_argument("action", choices=["submit", "poll", "wait"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--chunk", type=int, default=2000)
    p.add_argument("--all-stages", action="store_true", help="also judge wording-validation and follow-up records")
    p.add_argument("--interval", type=int, default=300)
    args = p.parse_args()
    exp = experiment_for(args)
    cfg = exp["judge"]
    if cfg["provider"] != "openrouter":
        raise SystemExit("batch judging is implemented for judge.provider = openrouter")
    key = args.model
    if args.action == "submit":
        pending = {it["id"] for it in load_state(key)["items"].values()}
        pairs = [(c, r) for c in conditions_present(key) for r in roles_present(key, c)]
        ws = ("discovery", "none", "validation", "followup") if args.all_stages else ("discovery", "none")
        items, n_skip = collect_items(key, pairs, exp["default_role"], ws, exclude_ids=pending)
        if args.limit:
            items = items[: args.limit]
        print(f"[judge-batch] {len(items)} to judge, {n_skip} skipped (default role / no answer), "
              f"{len(pending)} already submitted", flush=True)
        if items:
            submit(key, items, cfg, chunk=args.chunk)
    elif args.action == "poll":
        print(json.dumps(poll(key, cfg)), flush=True)
    else:
        while True:
            c = poll(key, cfg)
            print(time.strftime("%H:%M:%S"), json.dumps(c), flush=True)
            if c["pending"] == 0:
                n = resubmit_unparsed(key, cfg)
                if n == 0:
                    break
                print(f"[judge-batch] resubmitted {n} unparseable verdicts", flush=True)
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
