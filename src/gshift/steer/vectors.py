"""The steering set for one model: our directions, Betley-style directions, random controls.

Every direction is stored as a UNIT vector at our layer convention (input of block L);
the applied vector is strength x A_L x norm_factor x unit, with A_L the model's mean
residual norm at L (Betley et al.'s scale). norm_factor is 1 for every direction except
the original-contribution decomposition components, so all other directions are
norm-matched at a given strength.

Decomposition (configs/steering.yaml `projections`): for the discovery-only orthonormal
baseline PCA basis U_k of the source direction's group / representation / layer,
g_par = U_k U_k^T g and g_perp = g - g_par. "<src>_in_k<k>" / "<src>_out_k<k>" steer
along unit(g_par) / unit(g_perp) with norm_factor ||component|| / ||g||, i.e. exactly the
component's share of strength x A_L x g/||g|| (never renormalised); "..._mn" are the same
directions at matched norm (norm_factor 1), skipped when the component is numerically
negligible (||component|| / ||g|| < min_component_frac). Ranks are capped at the PCA's
empirical rank. "Inside/outside the baseline PCA subspace" is the only claim: a residual
outside this sampled subspace is not thereby outside all persona representations.
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
            if d.get("orthogonal_to"):
                v, removed = orthogonalise(v, [np.load(adir / f"{o['group']}_{o['rep']}_L{L}_{o['contrast']}.npy")
                                               .astype(np.float64) for o in d["orthogonal_to"]])
                src += " minus its projection on " + ", ".join(
                    f"{o['group']}_{o['rep']}_L{L}_{o['contrast']}" for o in d["orthogonal_to"])
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
        if d.get("orthogonal_to"):
            manifest["directions"][name]["removed_norm_frac"] = removed
    raw: dict[str, np.ndarray] = {}
    for name, d in cfg["directions"].items():
        if name in manifest["directions"] and d["source"] == "analysis" and not d.get("orthogonal_to"):
            raw[name] = np.load(adir / f"{d['group']}_{d['rep']}_L{L}_{d['contrast']}.npy").astype(np.float64)
    add_projections(manifest, cfg, raw, adir, L, out)
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


def orthogonalise(v: np.ndarray, others: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """v minus its projection on span(others); also the removed fraction of ||v||^2."""
    Q, _ = np.linalg.qr(np.stack(others, axis=1))
    r = v - Q @ (Q.T @ v)
    return r, float(1.0 - (r @ r) / (v @ v))


def add_projections(manifest: dict[str, Any], cfg: dict[str, Any], raw: dict[str, np.ndarray], adir: Path, L: int,
                    out: Path) -> None:
    """Inside / outside the baseline PCA subspace components of the source directions."""
    pc = cfg.get("projections") or {}
    manifest.setdefault("skipped", {})
    min_frac = float(pc.get("min_component_frac", 0.1))
    for src in pc.get("sources", []):
        if src not in raw:
            manifest["skipped"][src] = "projection source is not an analysis direction of this model"
            continue
        d = cfg["directions"][src]
        g = raw[src]
        gn = float(np.linalg.norm(g))
        V = np.load(adir / f"{d['group']}_{d['rep']}_L{L}_baseline_pcs.npy").astype(np.float64)
        done = set()
        for k in pc.get("ranks", []):
            kk = min(int(k), len(V))
            if kk in done or kk == 0:
                continue
            done.add(kk)
            par = V[:kk].T @ (V[:kk] @ g)
            for side, comp in (("in", par), ("out", g - par)):
                frac = float(np.linalg.norm(comp) / gn)
                name = f"{src}_{side}_k{kk}"
                label = (f"{'inside' if side == 'in' else 'outside'} the rank-{kk} baseline PCA subspace "
                         f"({d['group']}/{d['rep']}, discovery roles)")
                if frac < 1e-6:
                    manifest["skipped"][name] = f"component norm {frac:.2e} x ||g||: nothing to steer along"
                    continue
                u = _unit(comp)
                common = {"kind": "projection", "projection_of": src, "rank": kk, "requested_rank": int(k),
                          "side": side, "component_norm_frac": frac, "source": f"{src} decomposed on "
                          f"analysis/vectors/{d['group']}_{d['rep']}_L{L}_baseline_pcs.npy"}
                np.save(out / f"{name}.npy", u)
                manifest["directions"][name] = {**common, "norm_factor": frac,
                                                "label": label + ", original contribution"}
                if frac < min_frac:
                    manifest["skipped"][name + "_mn"] = (f"matched-norm version not run: component is "
                                                         f"{frac:.3f} < {min_frac} of ||g|| (not renormalised)")
                    continue
                np.save(out / f"{name}_mn.npy", u)
                manifest["directions"][name + "_mn"] = {**common, "norm_factor": 1.0,
                                                        "label": label + ", matched norm"}


def direction_scale(manifest: dict[str, Any], direction: str, signed_strength: float) -> float:
    """Signed scale of the applied vector: strength x A_L x norm_factor (x unit direction)."""
    nf = float(manifest.get("directions", {}).get(direction, {}).get("norm_factor", 1.0))
    return float(signed_strength) * float(manifest["activation_norm"]) * nf


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
