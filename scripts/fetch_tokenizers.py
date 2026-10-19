#!/usr/bin/env python3
"""Cache both models' tokenizer/config files (a few MB, no weights) for the offline tests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import snapshot_download  # noqa: E402

from gshift.config import load_models  # noqa: E402

for spec in load_models().values():
    p = snapshot_download(spec.hf_id, revision=spec.revision,
                          allow_patterns=["*.json", "*.jinja", "merges.txt", "vocab.json"])
    print(spec.hf_id, "->", p)
