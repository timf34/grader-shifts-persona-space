#!/usr/bin/env python3
"""Stage 3 (CPU + API): the existing Assistant Axis role-adherence judge, blind to condition.
Runs on the pod or a laptop (needs OPENAI_API_KEY, or ANTHROPIC_API_KEY if provider=anthropic).

    python -u scripts/03_judge.py --model qwen3.6-27b
    python -u scripts/03_judge.py --model qwen3.6-27b --conditions baseline auto human --limit 100
"""

from _common import base_parser, experiment_for

from gshift.io import conditions_present, roles_present
from gshift.judge import run_judge


def main():
    p = base_parser(__doc__)
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    exp = experiment_for(args)
    pairs = [(c, r) for c in (args.conditions or conditions_present(args.model)) for r in roles_present(args.model, c)]
    run_judge(args.model, pairs, exp["judge"], exp["default_role"], args.limit)


if __name__ == "__main__":
    main()
