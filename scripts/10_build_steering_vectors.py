#!/usr/bin/env python3
"""Q2 step 1 (CPU): assemble this model's steering set from its own Q1 outputs.

    python -u scripts/10_build_steering_vectors.py --model qwen3.6-27b

Needs outputs/<model>/analysis/vectors (04_analyze.py) and betley_repro (02_extract.py).
Writes outputs/<model>/steering/{vectors/*.npy, vectors.json} (unit vectors + scale A_L).
"""

import json

from _common import ROOT, base_parser, experiment_for

from gshift.config import load_yaml
from gshift.steer.vectors import build_steering_set


def main():
    p = base_parser(__doc__)
    p.add_argument("--steering-config", default=str(ROOT / "configs" / "steering.yaml"))
    args = p.parse_args()
    experiment_for(args)   # sets the smoke output root if --smoke
    man = build_steering_set(args.model, load_yaml(args.steering_config))
    print(json.dumps({k: v for k, v in man.items() if k != "cosine_matrix"}, indent=1))
    print("cosines between directions:", json.dumps(man["cosine_matrix"]))


if __name__ == "__main__":
    main()
