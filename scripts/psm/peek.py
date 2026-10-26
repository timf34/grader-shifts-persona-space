#!/usr/bin/env python3
"""Quick look at a model's PSM generations (local copy): validity, outcome mix and misreport rates
by persona x context, by pressure, and the steering sweep table.

    python scripts/psm/peek.py outputs/<model>/psm
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from gshift.psm.score import load_scored  # noqa: E402


def main():
    d = Path(sys.argv[1])
    rows = load_scored(d / "gen")
    print(f"{len(rows)} rows from {d}")
    rpt = [r for r in rows if r["task"] == "rpt" and r["benchmark"].startswith("rpt_")]
    if rpt:
        print("\n== Track A report: outcome mix by context (conflict items)")
        for c in sorted({r["context"] for r in rpt}):
            conf = [r for r in rpt if r["context"] == c and r["kind"] == "conflict"]
            ctrl = [r for r in rpt if r["context"] == c and r["kind"] != "conflict"]
            oc = Counter(r["outcome"] for r in conf)
            print(f"  {c:8} n={len(conf)} " + " ".join(f"{k}={v / len(conf):.2f}" for k, v in sorted(oc.items())) +
                  f" | control acc {np.mean([r['outcome'] == 'accurate' for r in ctrl]):.2f}"
                  f" | trunc {np.mean([r['truncated'] for r in conf]):.2f}")
        print("\n== misreport by pressure x context (valid conflict)")
        for pr in ("mild", "strong", "explicit"):
            line = []
            for c in sorted({r["context"] for r in rpt}):
                v = [r["misreport"] for r in rpt if r["context"] == c and r["pressure"] == pr and r["valid"]]
                line.append(f"{c} {np.mean(v):.2f}" if v else f"{c} -")
            print(f"  {pr:8} " + " | ".join(line))
        print("\n== misreport by persona x context (valid conflict)")
        ctxs = sorted({r["context"] for r in rpt})
        per = defaultdict(lambda: defaultdict(list))
        for r in rpt:
            if r["kind"] == "conflict" and r["valid"]:
                per[r["persona"]][r["context"]].append(r["misreport"])
        print("  " + " " * 16 + " ".join(f"{c:>8}" for c in ctxs))
        for p in per:
            print(f"  {p:16}" + " ".join(f"{np.mean(per[p][c]):8.2f}" if per[p][c] else "       -" for c in ctxs))
    q = [r for r in rows if r["task"] == "rptq"]
    if q:
        print("\n== questionnaire: choose favoured (valid) by context")
        for c in sorted({r["context"] for r in q}):
            v = [r["choose_favoured"] for r in q if r["context"] == c and r["valid"]]
            print(f"  {c:8} {np.mean(v):.2f} (valid {np.mean([r['valid'] for r in q if r['context'] == c]):.2f})")
    st = [r for r in rows if r["benchmark"].startswith("steer_")]
    if st:
        print("\n== steering (conflict misreport | valid | control acc | median answer tokens)")
        by = defaultdict(list)
        for r in st:
            by[(r["context"], r["condition"])].append(r)
        for (c, cond), rs in sorted(by.items()):
            conf = [r for r in rs if r["kind"] == "conflict" and r["valid"]]
            ctrl = [r for r in rs if r["kind"] != "conflict"]
            print(f"  {c:7} {cond:22} mis {np.mean([r['misreport'] for r in conf]) if conf else float('nan'):.2f} "
                  f"valid {np.mean([r['valid'] for r in rs]):.2f} ctrl {np.mean([r['outcome'] == 'accurate' for r in ctrl]):.2f} "
                  f"ans {np.median([r['n_answer_tokens'] or 0 for r in rs]):.0f} n={len(rs)}")


if __name__ == "__main__":
    main()
