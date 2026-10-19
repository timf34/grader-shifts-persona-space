#!/usr/bin/env python3
"""Stage 5 (CPU): Qwen3.6 vs Qwen3.8 comparison -> outputs/comparison/.

    python -u scripts/05_compare_models.py --a qwen3.6-27b --b qwen3.8-27b
"""

import argparse
import json

from _common import load_experiment

from gshift.analysis.crossmodel import compare_models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", default="qwen3.6-27b")
    p.add_argument("--b", default="qwen3.8-27b")
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    res = compare_models(args.a, args.b, load_experiment(), n_boot=args.n_boot)
    print(json.dumps(res["primary"], indent=1)[:4000])


if __name__ == "__main__":
    main()
