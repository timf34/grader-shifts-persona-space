#!/usr/bin/env python3
"""Push outputs/<model>/ to the HF dataset repo in $HF_DATASET_REPO (created private if missing).
Results live there, not on a network volume (runpod-runner storage policy).

    python -u scripts/upload_hf.py --model qwen3.6-27b --message "Q1 analysis"
    python -u scripts/upload_hf.py --comparison
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

from huggingface_hub import HfApi  # noqa: E402

from gshift.io import outputs_root  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=None)
    p.add_argument("--comparison", action="store_true")
    p.add_argument("--message", default="update outputs")
    p.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"))
    p.add_argument("--public", action="store_true", help="create the dataset public (default private)")
    args = p.parse_args()
    if not args.repo:
        sys.exit("set HF_DATASET_REPO (or --repo)")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    sub = "comparison" if args.comparison else args.model
    if not sub or not (outputs_root() / sub).exists():
        sys.exit(f"nothing to upload at {outputs_root() / str(sub)}")
    # upload_large_folder is resumable and handles the ~10 GB activation stores
    api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(outputs_root()),
                            allow_patterns=[f"{sub}/**"], ignore_patterns=["**/*.tmp"])
    print(f"[upload] {sub} -> https://huggingface.co/datasets/{args.repo} ({args.message})")


if __name__ == "__main__":
    main()
