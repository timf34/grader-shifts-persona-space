#!/usr/bin/env python3
"""Q2 step 3 (CPU + API): the model-graded Q2 metrics, resumable.
  * School of Reward Hacks: metric + quality judges (Betley et al.'s prompts) on every
    sorh / sorh_descored generation -> steering/sorh_judgements.jsonl
  * open-ended traits: the frozen-rubric trait judge on every trait_openended generation
    -> steering/trait_judgements.jsonl
  * persona adherence (adherence, adherence_graded): the Q1 judge -> steering/adherence_judgements.jsonl
  * agentic misalignment: the reference classifiers -> steering/agentic_judgements.jsonl
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
    p.add_argument("--only", choices=["sorh", "traits", "adherence", "agentic", "choice", "proxy"], default=None)
    args = p.parse_args()
    exp = experiment_for(args)
    cfg = load_yaml(args.steering_config)
    sd = steering_dir(args.model)
    if args.only in (None, "sorh"):
        files = sorted((sd / "gen" / "sorh").glob("*.jsonl")) + sorted((sd / "gen" / "sorh_descored").glob("*.jsonl"))
        if files:
            judge_sorh(files, sd / "sorh_judgements.jsonl", cfg["judges"])
    if args.only in (None, "traits"):
        files = sorted((sd / "gen" / "trait_openended").glob("*.jsonl"))
        if files:
            judge_traits(files, sd / "trait_judgements.jsonl", cfg["judges"])
    if args.only in (None, "adherence"):
        files = sorted((sd / "gen" / "adherence").glob("*.jsonl")) + sorted((sd / "gen" / "adherence_graded").glob("*.jsonl"))
        if files:
            from gshift.steer.adherence import judge_adherence

            judge_adherence(files, sd / "adherence_judgements.jsonl", exp["judge"])
    if args.only in (None, "choice"):
        # prose persona expression on free-format persona-choice answers: the Q1 judge, unchanged
        files = sorted(f for d in (sd / "gen").glob("choice_free_*") for f in d.glob("*.jsonl"))
        if files:
            from gshift.steer.persona_choice import judge_choice_adherence

            judge_choice_adherence(files, sd / "choice_adherence_judgements.jsonl", exp["judge"])
    if args.only in (None, "proxy"):
        files = sorted(f for d in (sd / "gen").glob("proxy_*") for f in d.glob("*.jsonl"))
        if files and cfg["judges"].get("proxy_quality_model"):
            from gshift.steer.proxy_panel import judge_proxy_quality

            judge_proxy_quality(files, sd / "proxy_quality_judgements.jsonl", cfg)
    if args.only in (None, "agentic"):
        files = sorted((sd / "gen" / "agentic").glob("*.jsonl"))
        if files:
            from gshift.steer.agentic import judge_agentic

            judge_agentic(files, sd / "agentic_judgements.jsonl", cfg)


if __name__ == "__main__":
    main()
