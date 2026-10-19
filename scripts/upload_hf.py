#!/usr/bin/env python3
"""Push results to the HF dataset repo in $HF_DATASET_REPO. Results live there, not on a
network volume (runpod-runner storage policy). The dataset is created PUBLIC if missing
(private datasets hit the storage quota mid-run); --private creates it private; an
existing repo's visibility is never changed. Every upload is VERIFIED: the remote file list under the
uploaded prefix must contain every local file, else exit 3 (the pod runner then keeps the
pod's disk instead of terminating it).

    python -u scripts/upload_hf.py --model qwen3.6-27b --message "Q1 analysis"      # outputs/<model>/
    python -u scripts/upload_hf.py --comparison
    python -u scripts/upload_hf.py --folder outputs_pilot/thinking_on/qwen3.6-27b \\
        --path-in-repo pilot/thinking_on/qwen3.6-27b --message "pilot"
    python -u scripts/upload_hf.py --folder /workspace/logs/q1-qwen3.6-27b --path-in-repo logs/q1/qwen3.6-27b
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

IGNORE = ["**/*.tmp", "**/__pycache__/**"]


def local_files(folder: Path) -> set[str]:
    return {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and not p.name.endswith(".tmp")
            and "__pycache__" not in p.parts}


def verify(api: HfApi, repo: str, folder: Path, prefix: str) -> None:
    remote = {f[len(prefix) + 1:] for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(prefix + "/")}
    missing = sorted(local_files(folder) - remote)
    if missing:
        print(f"!! [upload] {len(missing)} files missing on the hub under {prefix}/ (e.g. {missing[:3]})", flush=True)
        sys.exit(3)
    print(f"[upload] verified {len(remote)} files under {prefix}/", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None)
    p.add_argument("--comparison", action="store_true")
    p.add_argument("--folder", default=None, help="upload this local folder ...")
    p.add_argument("--path-in-repo", default=None, help="... to this path in the dataset repo")
    p.add_argument("--message", default="update outputs")
    p.add_argument("--repo", default=os.environ.get("HF_DATASET_REPO"))
    p.add_argument("--private", action="store_true", help="create the dataset private (default public)")
    args = p.parse_args()
    if not args.repo:
        sys.exit("set HF_DATASET_REPO (or --repo)")
    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=args.private, exist_ok=True)
    if not args.private and api.dataset_info(args.repo).private:
        print(f"!! [upload] {args.repo} already exists and is PRIVATE (left unchanged; private storage quota "
              "applies). Make it public on the hub if intended.", flush=True)
    if args.folder:
        folder = Path(args.folder)
        if not args.path_in_repo:
            sys.exit("--folder needs --path-in-repo")
        if not folder.is_dir() or not local_files(folder):
            sys.exit(f"nothing to upload at {folder}")
        prefix = args.path_in_repo.strip("/")
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(folder), path_in_repo=prefix,
                          commit_message=args.message, ignore_patterns=IGNORE)
        verify(api, args.repo, folder, prefix)
    else:
        sub = "comparison" if args.comparison else args.model
        if not sub or not (outputs_root() / sub).exists():
            sys.exit(f"nothing to upload at {outputs_root() / str(sub)}")
        # upload_large_folder is resumable and handles the ~10 GB activation stores
        api.upload_large_folder(repo_id=args.repo, repo_type="dataset", folder_path=str(outputs_root()),
                                allow_patterns=[f"{sub}/**"], ignore_patterns=IGNORE)
        verify(api, args.repo, outputs_root() / sub, sub)
        prefix = sub
    print(f"[upload] {prefix} -> https://huggingface.co/datasets/{args.repo} ({args.message})")


if __name__ == "__main__":
    main()
