#!/usr/bin/env python3
"""Q2 step 3 (CPU + API): School of Reward Hacks judges (metric + quality, Betley et al.'s
prompts, claude-sonnet-5) on every sorh / sorh_descored generation. Resumable.

    python -u scripts/12_steer_judge.py --model qwen3.6-27b
"""

from _common import ROOT, base_parser, experiment_for

from gshift.config import load_yaml
from gshift.steer.judges import judge_sorh
from gshift.steer.vectors import steering_dir


def main():
    p = base_parser(__doc__)
    p.add_argument("--steering-config", default=str(ROOT / "configs" / "steering.yaml"))
    args = p.parse_args()
    experiment_for(args)
    cfg = load_yaml(args.steering_config)
    sd = steering_dir(args.model)
    files = sorted((sd / "gen" / "sorh").glob("*.jsonl")) + sorted((sd / "gen" / "sorh_descored").glob("*.jsonl"))
    judge_sorh(files, sd / "sorh_judgements.jsonl", cfg["judges"])


if __name__ == "__main__":
    main()
