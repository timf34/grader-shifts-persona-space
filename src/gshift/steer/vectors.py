"""The steering set for one model: our directions, Betley-style directions, random controls.

Every direction is stored as a UNIT vector at our layer convention (input of block L);
the applied vector is strength x A_L x unit, with A_L the model's mean residual norm at L
(Betley et al.'s scale), so all directions are norm-matched at a given strength.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ..io import model_dir, write_json


def steering_dir(model_key: str) -> Path:
    p = model_dir(model_key) / "steering"
    p.mkdir(parents=True, exist_ok=True)
    return p


def activation_norm(model_key: str, layer: int) -> float:
    rep = model_dir(model_key) / "betley_repro" / "report.json"
    if not rep.exists():
        raise FileNotFoundError(f"{rep} missing: run scripts/02_extract.py (Betley reproduction) first")
    norms = json.loads(rep.read_text())["per_layer_mean_activation_norm"]
    return float(norms[layer])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n == 0:
        raise ValueError("zero or non-finite direction")
    return (v / n).astype(np.float32)


def build_steering_set(model_key: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Write steering/vectors/<name>.npy (unit vectors) and steering/vectors.json."""
    from ..betley import published

    L = int(cfg["layer"])
    out = steering_dir(model_key) / "vectors"
    out.mkdir(parents=True, exist_ok=True)
    adir = model_dir(model_key) / "analysis" / "vectors"
    manifest: dict[str, Any] = {"model_key": model_key, "layer": L,
                                "easysteer_layer": L - 1,
                                "layer_convention": "our layer L = input of block L = EasySteer output of block L-1",
                                "activation_norm": activation_norm(model_key, L), "directions": {}}
    H = None
    for name, d in cfg["directions"].items():
        if d.get("models") and model_key not in d["models"]:
            continue
        if d["source"] == "analysis":
            p = adir / f"{d['group']}_{d['rep']}_L{L}_{d['contrast']}.npy"
            if not p.exists():
                raise FileNotFoundError(f"{p} missing: run scripts/04_analyze.py first")
            v = np.load(p).astype(np.float64)
            src = str(p.relative_to(model_dir(model_key)))
        elif d["source"] == "betley_repro":
            v = np.load(model_dir(model_key) / "betley_repro" / "deltas_all_layers.npy")[L].astype(np.float64)
            src = "betley_repro/deltas_all_layers.npy"
        elif d["source"] == "betley_published":
            v = published()["deltas"][L].astype(np.float64)
            src = "data/betley_0007/deltas_all_layers.npy"
        else:
            raise ValueError(f"unknown direction source {d['source']}")
        u = _unit(v)
        H = u.size
        np.save(out / f"{name}.npy", u)
        manifest["directions"][name] = {"source": src, "raw_norm": float(np.linalg.norm(v)), "kind": "recovered"}
    rnd = cfg.get("random_directions", {})
    rng = np.random.default_rng(int(rnd.get("seed", 0)))
    for k in range(int(rnd.get("n", 0))):
        u = _unit(rng.normal(size=H))
        np.save(out / f"random{k}.npy", u)
        manifest["directions"][f"random{k}"] = {"source": f"isotropic normal, seed {rnd.get('seed')} #{k}",
                                                "kind": "random"}
    names = list(manifest["directions"])
    U = np.stack([np.load(out / f"{n}.npy") for n in names]).astype(np.float64)
    manifest["cosine_matrix"] = {"names": names, "cos": (U @ U.T).round(4).tolist()}
    write_gguf_files(model_key, manifest)
    write_json(steering_dir(model_key) / "vectors.json", manifest)
    return manifest


def write_gguf_files(model_key: str, manifest: dict[str, Any]) -> dict[str, Path]:
    """EasySteer `direct` payloads (unit vectors at EasySteer layer L-1). Needs `gguf`."""
    from .engine import write_direction_gguf

    sd = steering_dir(model_key)
    L = int(manifest["layer"])
    return {n: write_direction_gguf(np.load(sd / "vectors" / f"{n}.npy"), L, sd / "gguf" / f"{n}.gguf")
            for n in manifest["directions"]}


def gguf_paths(model_key: str, manifest: dict[str, Any]) -> dict[str, Path]:
    sd = steering_dir(model_key)
    paths = {n: sd / "gguf" / f"{n}.gguf" for n in manifest["directions"]}
    if not all(p.exists() for p in paths.values()):
        paths = write_gguf_files(model_key, manifest)
    return paths


def load_steering_set(model_key: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sd = steering_dir(model_key)
    man = json.loads((sd / "vectors.json").read_text())
    vecs = {n: np.load(sd / "vectors" / f"{n}.npy") for n in man["directions"]}
    return man, vecs
