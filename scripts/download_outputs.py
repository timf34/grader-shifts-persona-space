#!/usr/bin/env python3
"""Pull a model's outputs back from $HF_DATASET_REPO into outputs/ (e.g. on a fresh pod for
Q2, or on a laptop for analysis).

    python -u scripts/download_outputs.py --model qwen3.6-27b --what analysis betley_repro
    python -u scripts/download_outputs.py --model qwen3.6-27b --what all
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:
    pass

from huggingface_hub import snapshot_download  # noqa: E402

from gshift.io import outputs_root  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--what", nargs="+", default=["analysis", "betley_repro"],
                   help="subdirectories of outputs/<model>/ (or 'all')")
    p.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"))
    args = p.parse_args()
    if not args.repo:
        sys.exit("set HF_DATASET_REPO (or --repo)")
    pats = [f"{args.model}/**"] if "all" in args.what else [f"{args.model}/{w}/**" for w in args.what]
    snapshot_download(args.repo, repo_type="dataset", local_dir=str(outputs_root()), allow_patterns=pats)
    print(f"[download] {pats} -> {outputs_root()}")


if __name__ == "__main__":
    main()
