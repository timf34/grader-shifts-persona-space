"""Synthetic model outputs in the real on-disk layout, with planted structure:

  shift(r, cond) = s_g * u_grading                        (any grading statement)
                 + s_e * evaluator(cond) * u_evaluator    (+1 automated, -1 human)
                 + s_c * criterion(cond) * u_criterion    (+1 correctness, -1 quality)
                 - alpha * (persona_r - mean persona)     (convergence, if alpha > 0)
                 + role_specific * unit noise direction per (role, cond)
"""

from __future__ import annotations

import copy
import json

import numpy as np

from gshift.config import condition_def, load_experiment
from gshift.design import build_requests
from gshift.extract import META_FIELDS, REPS
from gshift.io import activations_dir, responses_path, scores_path

SMALL_ROLES = {
    "analytical_helping": ["librarian", "analyst", "researcher"],
    "relational": ["caregiver", "therapist", "counselor"],
    "competitive": ["negotiator", "lawyer", "competitor"],
    "creative": ["actor", "poet", "designer"],
    "nonhuman": ["robot", "alien", "ghost"],
}


def small_experiment() -> dict:
    """15 roles (5 held out) x 2 prompt variants x 60 questions."""
    e = copy.deepcopy(load_experiment())
    e["roles"] = SMALL_ROLES
    e["role_split"]["heldout"] = ["researcher", "counselor", "competitor", "designer", "ghost"]
    e["prompt_variants"] = [0, 1]
    e["stages"]["main"]["prompt_variants"] = [0, 1]
    e["analysis"].update(n_permutations=60, n_permutations_sweep=30, n_bootstrap=30, n_split_halves=4)
    e["analysis"]["geometry"].update(n_permutations=60, n_bootstrap=40)
    return e


def _unit(rng, H):
    v = rng.normal(0, 1, H)
    return v / np.linalg.norm(v)


def write_synthetic_model(model_key: str, exp: dict, stages=("main",), H: int = 48, layers=(16, 36),
                          s_g: float = 1.5, s_e: float = 1.0, s_c: float = 0.8, alpha: float = 0.0,
                          role_specific: float = 0.3, noise: float = 0.3, persona_scale: float = 3.0,
                          seed: int = 0, world_seed: int = 0, judge: bool = True) -> dict:
    world = np.random.default_rng(world_seed)
    rng = np.random.default_rng(seed + 1000)
    u_g, u_e, u_c = _unit(world, H), _unit(world, H), _unit(world, H)
    reqs = [r for st in stages for r in build_requests(exp, st)]
    roles = sorted({r.role for r in reqs})
    persona = {r: world.normal(0, persona_scale, H) for r in roles}
    pbar = np.mean([persona[r] for r in roles if r != exp["default_role"]], 0)
    qeff = {}
    spec = {}
    for r in reqs:
        qeff.setdefault(r.question_id, world.normal(0, 1, H))
        spec.setdefault((r.role, r.condition), _unit(world, H))
    files: dict[tuple[str, str], list] = {}
    for r in reqs:
        files.setdefault((r.condition, r.role), []).append(r)
    for (cond, role), rs in files.items():
        d = condition_def(exp, cond)
        ev = {"automated": 1.0, "human": -1.0}.get(d["evaluator"], 0.0)
        cr = {"correctness": 1.0, "quality": -1.0}.get(d["criterion"], 0.0)
        rows, idx, arrs, norms, scores = [], [], {k: [] for k in REPS}, [], []
        for i, r in enumerate(rs):
            shift = np.zeros(H)
            if cond != "baseline":
                shift = s_g * u_g + s_e * ev * u_e + s_c * cr * u_c + role_specific * spec[(role, cond)]
                if role != exp["default_role"]:
                    shift = shift - alpha * (persona[role] - pbar)
            base = persona[role] + qeff[r.question_id]
            vecs = {k: np.stack([base + shift + rng.normal(0, noise, H) for _ in layers]) for k in REPS}
            for k in REPS:
                arrs[k].append(vecs[k])
            norms.append(np.linalg.norm(np.stack([vecs[k] for k in REPS]), axis=-1))
            has = bool(rng.random() > 0.02)
            dd = r.to_dict()
            rows.append({**dd, "config_id": "synthetic", "has_answer": has, "truncated": not has,
                         "finish_reason": "stop" if has else "length", "answer_text": "x",
                         "gen_ids": [0] * 160, "max_tokens": 8192, "think_closed": has,
                         "n_reasoning_tokens": 100, "n_answer_tokens": 50 if has else 0})
            idx.append({"id": r.id, "row": i, "config_id": "synthetic", **{k: dd.get(k) for k in META_FIELDS},
                        "n_input_tokens": 200, "n_reasoning_tokens": 100, "n_answer_tokens": 50 if has else 0,
                        "think_closed": has, "has_answer": has, "n_generated": 160, "truncated": not has,
                        "finish_reason": "stop" if has else "length"})
            if judge and role != exp["default_role"]:
                scores.append({"id": r.id, "role": role, "condition": cond, "score": int(rng.choice([2, 3, 3, 3]))})
        rp = responses_path(model_key, cond, role)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text("".join(json.dumps(x) + "\n" for x in rows))
        ad = activations_dir(model_key, cond, role)
        ad.mkdir(parents=True, exist_ok=True)
        for k in REPS:
            np.save(ad / f"{k}.npy", np.stack(arrs[k]).astype(np.float16))
        np.save(ad / "norms.npy", np.stack(norms).astype(np.float32))
        (ad / "meta.json").write_text(json.dumps({"layers": list(layers)}))
        (ad / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in idx))
        if scores:
            sp = scores_path(model_key, cond, role)
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_text("".join(json.dumps(x) + "\n" for x in scores))
    return {"u_g": u_g, "u_e": u_e, "u_c": u_c}
