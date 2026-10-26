#!/usr/bin/env python3
"""Stage 5 (CPU): cross-model comparison (default Qwen3.6 vs Qwen3.8) -> outputs/comparison/ or --out-dir.

    python -u scripts/05_compare_models.py --a qwen3.6-27b --b qwen3.8-27b
    python -u scripts/05_compare_models.py --a qwen3.5-27b --b qwen3.6-27b --out-dir outputs/comparison_qwen3.5_vs_qwen3.6
"""

import argparse
import json
from pathlib import Path

from _common import load_experiment

from gshift.analysis.crossmodel import compare_models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--a", default="qwen3.6-27b")
    p.add_argument("--b", default="qwen3.8-27b")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--out-dir", type=Path, default=None, help="default <outputs>/comparison")
    args = p.parse_args()
    res = compare_models(args.a, args.b, load_experiment(), out_dir=args.out_dir, n_boot=args.n_boot)
    print(json.dumps(res["primary"], indent=1)[:4000])


if __name__ == "__main__":
    main()
