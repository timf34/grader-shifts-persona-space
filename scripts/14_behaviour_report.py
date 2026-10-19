#!/usr/bin/env python3
"""Q2 step 5 (CPU): behavioural summary -> outputs/<model>/steering/{behaviour_summary.json, tables/, figures/}

    python -u scripts/14_behaviour_report.py --model qwen3.6-27b
"""

from _common import base_parser, experiment_for

from gshift.steer.report import behaviour_summary
from gshift.steer.vectors import steering_dir


def main():
    p = base_parser(__doc__)
    args = p.parse_args()
    experiment_for(args)
    s = behaviour_summary(args.model, steering_dir(args.model))
    print(f"[behaviour] {s['n_samples']} samples, {len(s['conditions'])} conditions")


if __name__ == "__main__":
    main()
