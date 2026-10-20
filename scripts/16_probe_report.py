#!/usr/bin/env python3
"""Context-probe report (CPU, laptop-safe: one condition's layer-36 slice in memory at a
time, ~150 MB peak) -> outputs/<model>/probes/report.{json,md}.

    python scripts/16_probe_report.py --model qwen3.6-27b
    python scripts/16_probe_report.py --model qwen3.6-27b --smoke        # outputs_smoke/

Needs outputs/<model>/probes/ (scripts/15_context_probes.py) and, for the Q1 comparisons,
outputs/<model>/analysis/vectors/ (missing Q1 vectors are skipped, not faked).
"""

import argparse
import os

from _common import ROOT

from gshift.analysis.probes_analysis import probe_report
from gshift.io import write_json
from gshift.probes import load_probe_config, probes_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--probes-config", default=str(ROOT / "configs" / "probes.yaml"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n-bootstrap", type=int, default=None)
    args = p.parse_args()
    if args.smoke:
        os.environ["GSHIFT_OUTPUTS"] = os.environ.get("GSHIFT_SMOKE_OUTPUTS", str(ROOT / "outputs_smoke"))
    pcfg = load_probe_config(args.probes_config)
    if args.n_bootstrap is not None:
        pcfg.setdefault("analysis", {})["n_bootstrap"] = args.n_bootstrap
    rep, md = probe_report(args.model, pcfg)
    out = probes_dir(args.model)
    write_json(out / "report.json", rep)
    (out / "report.md").write_text(md)
    print(md)
    print(f"-> {out / 'report.json'}, {out / 'report.md'}")


if __name__ == "__main__":
    main()
