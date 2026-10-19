#!/usr/bin/env python3
"""Q2 step 3 (CPU + API): the model-graded Q2 metrics, resumable.
  * School of Reward Hacks: metric + quality judges (Betley et al.'s prompts) on every
    sorh / sorh_descored generation -> steering/sorh_judgements.jsonl
  * open-ended traits: the frozen-rubric trait judge on every trait_openended generation
    -> steering/trait_judgements.jsonl
TruthfulQA and GSM8K are scored locally (steer/report.py), no judge.

    python -u scripts/12_steer_judge.py --model qwen3.6-27b
    python -u scripts/12_steer_judge.py --model qwen3.6-27b --only traits
"""

from _common import ROOT, base_parser, experiment_for

from gshift.config import load_yaml
from gshift.steer.judges import judge_sorh
from gshift.steer.trait_judge import judge_traits
from gshift.steer.vectors import steering_dir


def main():
    p = base_parser(__doc__)
    p.add_argument("--steering-config", default=str(ROOT / "configs" / "steering.yaml"))
    p.add_argument("--only", choices=["sorh", "traits"], default=None)
    args = p.parse_args()
    experiment_for(args)
    cfg = load_yaml(args.steering_config)
    sd = steering_dir(args.model)
    if args.only in (None, "sorh"):
        files = sorted((sd / "gen" / "sorh").glob("*.jsonl")) + sorted((sd / "gen" / "sorh_descored").glob("*.jsonl"))
        judge_sorh(files, sd / "sorh_judgements.jsonl", cfg["judges"])
    if args.only in (None, "traits"):
        files = sorted((sd / "gen" / "trait_openended").glob("*.jsonl"))
        if files:
            judge_traits(files, sd / "trait_judgements.jsonl", cfg["judges"])


if __name__ == "__main__":
    main()
