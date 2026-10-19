"""Loading the experiment design (configs/experiment.yaml), model specs (configs/models.yaml)
and the versioned question manifest."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import CONFIG_DIR, DATA_DIR, REPO_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_experiment(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "experiment.yaml")


@dataclass(frozen=True)
class ModelSpec:
    key: str                     # short name used in output paths, e.g. "qwen3.6-27b"
    hf_id: str                   # e.g. "Qwen/Qwen3.6-27B"
    revision: str | None         # pinned commit sha (None = unpinned; refused for real runs)
    tokenizer_revision: str | None
    n_layers: int | None         # expected decoder blocks (checked at load time)
    hidden_size: int | None      # expected residual width (checked at load time)
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 128
    trust_remote_code: bool = False
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    vllm_kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


def load_models(path: str | Path | None = None) -> dict[str, ModelSpec]:
    raw = load_yaml(path or CONFIG_DIR / "models.yaml")
    out = {}
    for key, m in raw["models"].items():
        m = dict(m)
        known = {k: m.pop(k) for k in list(m) if k in ModelSpec.__dataclass_fields__}
        for k in ("chat_template_kwargs", "vllm_kwargs"):
            known[k] = dict(known.get(k) or {})
        out[key] = ModelSpec(key=key, extra=m, **known)
    return out


def get_model(key: str, path: str | Path | None = None) -> ModelSpec:
    models = load_models(path)
    if key not in models:
        raise KeyError(f"unknown model key {key!r}; known: {sorted(models)}")
    return models[key]


def require_pinned(spec: ModelSpec) -> None:
    """Refuse to run a real job on an unpinned model unless explicitly allowed."""
    if spec.revision is None and os.environ.get("GSHIFT_ALLOW_UNPINNED") != "1":
        raise RuntimeError(
            f"{spec.hf_id} has no pinned revision in configs/models.yaml. Run "
            "`python scripts/pin_revisions.py` or set GSHIFT_ALLOW_UNPINNED=1 for a smoke test."
        )


# ---------------------------------------------------------------------------
# data files
# ---------------------------------------------------------------------------

def load_role(name: str) -> dict[str, Any]:
    with open(DATA_DIR / "roles" / f"{name}.json") as f:
        return json.load(f)


def role_categories(exp: dict[str, Any]) -> dict[str, str]:
    """role -> category for the prompted roles, in config order (default excluded)."""
    return {r: cat for cat, roles in exp["roles"].items() for r in roles}


def load_questions(exp: dict[str, Any]) -> list[dict[str, Any]]:
    """The versioned manifest rows (id, text, source, group, domain, family, split, ...)."""
    path = Path(exp["question_manifest"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if "question_filter" in exp:   # smoke runs restrict to a subset of ids
        keep = set(exp["question_filter"])
        rows = [r for r in rows if r["id"] in keep]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate question ids in {path}")
    return rows


def manifest_sha(exp: dict[str, Any]) -> str:
    path = Path(exp["question_manifest"])
    if not path.is_absolute():
        path = REPO_ROOT / path
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def condition_texts(exp: dict[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for key in ("conditions", "paraphrase_conditions", "followup_conditions"):
        for name, text in (exp.get(key) or {}).items():
            if name in out and out[name] != text:
                raise ValueError(f"condition {name!r} defined twice with different text")
            out[name] = text
    return out
