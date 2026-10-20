#!/usr/bin/env python3
"""Stage 4 (CPU): Q1 analysis for one model -> outputs/<model>/analysis/.

    python -u scripts/04_analyze.py --model qwen3.6-27b
    python -u scripts/04_analyze.py --model qwen3.6-27b --reps resp pre --layers 36 --groups pooled --no-plots

Memory: one (representation, layer) slice at a time (~1-2 GB for the full grid).
"""

from _common import base_parser, experiment_for

from gshift.analysis.report import analyze_model


def main():
    p = base_parser(__doc__)
    p.add_argument("--reps", nargs="+", default=["pre", "resp", "reason"])
    p.add_argument("--layers", nargs="+", type=int, default=None)
    p.add_argument("--groups", nargs="+", default=None, help="original checkable pooled")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--out-dir", default=None,
                   help="write here instead of outputs/<model>/analysis (e.g. one directory per representation "
                        "run in parallel, then scripts/merge_analysis.py)")
    args = p.parse_args()
    exp = experiment_for(args)
    from pathlib import Path

    analyze_model(args.model, exp, out_dir=Path(args.out_dir) if args.out_dir else None, reps=tuple(args.reps),
                  layers=args.layers, groups=args.groups, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
