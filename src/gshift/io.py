"""Small IO helpers: jsonl, run metadata, output paths."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

from . import REPO_ROOT


def outputs_root() -> Path:
    return Path(os.environ.get("GSHIFT_OUTPUTS", REPO_ROOT / "outputs"))


# Layout (one directory per model x condition, one file per role, as in assistant-axis):
#   outputs/<model>/<condition>/responses/<role>.jsonl
#   outputs/<model>/<condition>/activations/<role>/{pre,resp,reason}.npy + index.jsonl + norms.npy
#   outputs/<model>/<condition>/scores/<role>.jsonl
#   outputs/<model>/<condition>/vectors/<role>.npz
#   outputs/<model>/run_config.json, betley_repro/, analysis/, steering/

def model_dir(model_key: str) -> Path:
    return outputs_root() / model_key


def cond_dir(model_key: str, condition: str) -> Path:
    return outputs_root() / model_key / condition


def responses_path(model_key: str, condition: str, role: str) -> Path:
    return cond_dir(model_key, condition) / "responses" / f"{role}.jsonl"


def activations_dir(model_key: str, condition: str, role: str) -> Path:
    return cond_dir(model_key, condition) / "activations" / role


def scores_path(model_key: str, condition: str, role: str) -> Path:
    return cond_dir(model_key, condition) / "scores" / f"{role}.jsonl"


def vectors_path(model_key: str, condition: str, role: str) -> Path:
    return cond_dir(model_key, condition) / "vectors" / f"{role}.npz"


def conditions_present(model_key: str) -> list[str]:
    d = model_dir(model_key)
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if (p / "responses").is_dir())


def roles_present(model_key: str, condition: str) -> list[str]:
    d = cond_dir(model_key, condition) / "responses"
    return sorted(p.stem for p in d.glob("*.jsonl")) if d.exists() else []


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Rows of a JSONL file. Lines that do not parse are SKIPPED with a warning (seen 2026-10-26 on a
    MooseFS/FUSE /workspace: a write left a run of NUL bytes that destroyed one record); a resumed
    run regenerates the missing ids."""
    path = Path(path)
    if not path.exists():
        return []
    out, bad = [], 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.replace(b"\x00", b"").strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                bad += 1
    if bad:
        import warnings
        warnings.warn(f"read_jsonl: skipped {bad} unparseable line(s) in {path}")
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
        f.flush()
        os.fsync(f.fileno())
    return n


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def git_commit() -> str | None:
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "-C", str(REPO_ROOT), "diff", "--quiet"],
                                stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "python": platform.python_version(),
        "repo_commit": git_commit(),
        "pod_id": os.environ.get("RUNPOD_POD_ID"),
    }
    for mod in ("torch", "transformers", "vllm", "numpy", "easysteer", "fla"):
        try:
            m = __import__(mod)
            info[f"{mod}_version"] = getattr(m, "__version__", "?")
        except Exception:
            info[f"{mod}_version"] = None
    try:
        import torch

        if torch.cuda.is_available():
            info["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            info["cuda"] = torch.version.cuda
    except Exception:
        pass
    return info
