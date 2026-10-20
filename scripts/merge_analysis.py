#!/usr/bin/env python3
"""Merge analysis runs made in parallel (one per representation, scripts/04_analyze.py
--reps <rep> --out-dir <part>) into outputs/<model>/analysis/: summary.json results
are disjoint by representation; vectors/ and figures/ file names carry the
representation; tables/layer_sweep.csv is concatenated.

    python scripts/merge_analysis.py --model qwen3.8-27b --parts outputs/qwen3.8-27b/analysis_parts/*
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gshift.io import outputs_root, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--parts", nargs="+", required=True)
    args = p.parse_args()
    out = outputs_root() / args.model / "analysis"
    for sub in ("vectors", "tables", "figures"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    merged, sweeps = None, []
    for part in map(Path, args.parts):
        s = json.loads((part / "summary.json").read_text())
        if merged is None:
            merged = s
        else:
            clash = set(merged["results"]) & set(s["results"])
            if clash:
                sys.exit(f"{part}: representations {sorted(clash)} already merged from another part")
            merged["results"].update(s["results"])
        for sub in ("vectors", "figures"):
            for f in (part / sub).glob("*"):
                shutil.copy2(f, out / sub / f.name)
        sw = part / "tables" / "layer_sweep.csv"
        if sw.exists() and sw.stat().st_size:
            sweeps.append(pd.read_csv(sw))
    merged["merged_from"] = [str(x) for x in args.parts]
    write_json(out / "summary.json", merged)
    if sweeps:
        pd.concat(sweeps, ignore_index=True).to_csv(out / "tables" / "layer_sweep.csv", index=False)
    print(f"merged {len(args.parts)} parts -> {out} (representations: {sorted(merged['results'])})")


if __name__ == "__main__":
    main()
