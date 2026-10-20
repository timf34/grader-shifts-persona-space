#!/usr/bin/env python3
"""Stage B: build the frozen shortcut panel from MBPP+ (evalplus/mbppplus at a pinned revision)
-> data/followup/shortcut_panel_v1.jsonl + PROVENANCE.json. Executes the dataset's own
reference solutions and hidden harnesses locally (trusted, pinned data) for the audit.

    python scripts/followup/b_build_panel.py --parquet <path to the pinned parquet>
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gshift.followup.shortcut_tasks import (CONDITIONS, N_DISCOVERY, N_TASKS, PANEL_SEED, build_candidate,  # noqa: E402
                                             panel_sha, select_panel)

REVISION = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parquet", required=True)
    args = p.parse_args()
    import pandas as pd

    raw = Path(args.parquet).read_bytes()
    df = pd.read_parquet(args.parquet)
    cands, reasons = [], collections.Counter()
    for row in df.to_dict(orient="records"):
        t, why = build_candidate(row)
        reasons[why] += 1
        if t:
            cands.append(t)
    print(f"{len(df)} MBPP+ tasks -> {len(cands)} eligible; {dict(reasons)}")
    panel, log = select_panel(cands)
    out = ROOT / "data" / "followup"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "shortcut_panel_v1.jsonl", "w") as f:
        for t in panel:
            f.write(json.dumps(t) + "\n")
    prov = {"source": "https://huggingface.co/datasets/evalplus/mbppplus", "revision": REVISION,
            "license": "apache-2.0", "parquet_sha256": hashlib.sha256(raw).hexdigest(),
            "n_source": len(df), "n_eligible": len(cands), "eligibility_reasons": dict(reasons),
            "panel_seed": PANEL_SEED, "n_tasks": N_TASKS, "n_discovery": N_DISCOVERY,
            "audit_visited": len(log), "audit_failed": [x for x in log if not x["ok"]],
            "panel_sha256": panel_sha(panel), "conditions": CONDITIONS,
            "construction": "last visible test's expected output corrupted (int+1, bool negated, str last char "
                            "changed, list/tuple last element corrupted); hidden = EvalPlus harness"}
    (out / "PROVENANCE_shortcut_panel_v1.json").write_text(json.dumps(prov, indent=1))
    print(json.dumps({k: v for k, v in prov.items() if k not in ("audit_failed", "conditions")}, indent=1))
    print("splits:", collections.Counter(t["split"] for t in panel))


if __name__ == "__main__":
    main()
