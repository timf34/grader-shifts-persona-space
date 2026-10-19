"""Analysis-side view of one model's outputs: record metadata (from the small activation
index files + judge scores) and memory-mapped activation arrays, one (rep, layer) at a time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..io import activations_dir, conditions_present, cond_dir, read_jsonl, scores_path


@dataclass
class Store:
    model_key: str
    meta: pd.DataFrame          # one row per extracted record (all conditions/roles)
    layers: list[int]

    @classmethod
    def load(cls, model_key: str) -> "Store":
        rows, layers = [], None
        for cond in conditions_present(model_key):
            adir = cond_dir(model_key, cond) / "activations"
            if not adir.exists():
                continue
            for rdir in sorted(p for p in adir.iterdir() if (p / "index.jsonl").exists()):
                m = json.loads((rdir / "meta.json").read_text())
                if layers is None:
                    layers = list(m["layers"])
                elif list(m["layers"]) != layers:
                    raise RuntimeError(f"{rdir}: captured at layers {m['layers']}, others at {layers}")
                scores = {r["id"]: r.get("score") for r in read_jsonl(scores_path(model_key, cond, rdir.name))}
                for r in read_jsonl(rdir / "index.jsonl"):
                    r["file_cond"], r["file_role"] = cond, rdir.name
                    r["score"] = scores.get(r["id"])
                    rows.append(r)
        if not rows:
            raise FileNotFoundError(f"no extracted activations for {model_key}")
        meta = pd.DataFrame(rows)
        for c in ("has_answer", "truncated", "think_closed"):
            meta[c] = meta[c].fillna(False).astype(bool)
        meta["score"] = pd.to_numeric(meta["score"], errors="coerce")
        return cls(model_key, meta, layers or [])

    def valid(self, rep: str, meta: pd.DataFrame | None = None) -> np.ndarray:
        m = self.meta if meta is None else meta
        if rep == "pre":
            return np.ones(len(m), bool)
        if rep == "resp":
            return (m["has_answer"] & (m["n_answer_tokens"].fillna(0) > 0)).to_numpy(bool)
        if rep == "reason":
            return (m["n_reasoning_tokens"].fillna(0) > 0).to_numpy(bool)
        raise KeyError(rep)

    def rows(self, rep: str, layer: int, meta: pd.DataFrame) -> np.ndarray:
        """Activation rows [len(meta), H] (float32) for the given metadata rows."""
        li = self.layers.index(layer)
        out = None
        for (cond, role), sub in meta.groupby(["file_cond", "file_role"], sort=False):
            arr = np.load(activations_dir(self.model_key, cond, role) / f"{rep}.npy", mmap_mode="r")
            block = np.asarray(arr[sub["row"].to_numpy(), li, :], dtype=np.float32)
            if out is None:
                out = np.zeros((len(meta), arr.shape[-1]), dtype=np.float32)
            out[meta.index.get_indexer(sub.index)] = block
        return out if out is not None else np.zeros((0, 0), np.float32)

    def norms(self, rep: str, layer: int, meta: pd.DataFrame) -> np.ndarray:
        li = self.layers.index(layer)
        k = ("pre", "resp", "reason").index(rep)
        out = np.zeros(len(meta), dtype=np.float32)
        for (cond, role), sub in meta.groupby(["file_cond", "file_role"], sort=False):
            n = np.load(activations_dir(self.model_key, cond, role) / "norms.npy", mmap_mode="r")
            out[meta.index.get_indexer(sub.index)] = n[sub["row"].to_numpy(), k, li]
        return out
