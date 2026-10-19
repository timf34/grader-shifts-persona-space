"""Synthetic model outputs with a planted shared shift, written in the real on-disk layout
(responses / activations / scores per condition and role)."""

from __future__ import annotations

import copy
import json

import numpy as np

from gshift.config import load_experiment
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
    """Reduced grid for tests: 15 roles x 2 prompt variants x 60 questions."""
    e = copy.deepcopy(load_experiment())
    e["roles"] = SMALL_ROLES
    e["role_split"]["heldout"] = ["researcher", "counselor", "competitor", "designer", "ghost"]
    e["prompt_variants"] = [0, 1]
    e["stages"]["main"]["prompt_variants"] = [0, 1]
    e["analysis"].update(n_permutations=80, n_permutations_sweep=40, n_bootstrap=40, n_split_halves=5)
    return e


def write_synthetic_model(model_key: str, exp: dict, stages=("main",), H: int = 48, layers=(16, 36),
                          shift_scale: float = 1.5, role_specific: float = 0.3, noise: float = 0.3,
                          seed: int = 0, world_seed: int = 0, judge: bool = True,
                          checkable_extra: float = 0.0) -> dict:
    world = np.random.default_rng(world_seed)
    rng = np.random.default_rng(seed + 1000)
    sh = world.normal(0, 1, H)
    shared = sh / np.linalg.norm(sh)
    extra = world.normal(0, 1, H)
    extra -= extra @ shared * shared
    extra /= np.linalg.norm(extra)
    persona, qeff, spec = {}, {}, {}
    reqs = [r for st in stages for r in build_requests(exp, st)]
    for r in reqs:
        if r.role not in persona:
            persona[r.role] = world.normal(0, 3, H)
        if r.question_id not in qeff:
            qeff[r.question_id] = world.normal(0, 1, H)
        if (r.role, r.condition) not in spec:
            v = world.normal(0, 1, H)
            spec[(r.role, r.condition)] = v / np.linalg.norm(v)
    files: dict[tuple[str, str], list] = {}
    for r in reqs:
        files.setdefault((r.condition, r.role), []).append(r)
    for (cond, role), rs in files.items():
        rows, idx, arrs, norms, scores = [], [], {k: [] for k in REPS}, [], []
        for i, r in enumerate(rs):
            base = persona[r.role] + qeff[r.question_id]
            shift = np.zeros(H)
            if cond.startswith("auto") or cond == "eval_awareness":
                sign = 1.0
            elif cond.startswith("human"):
                sign = -0.5
            else:
                sign = 0.0
            if sign:
                shift = sign * shift_scale * shared + role_specific * spec[(role, cond)]
                if r.question_group == "checkable":
                    shift = shift + sign * checkable_extra * extra
            vecs = {k: np.stack([base + shift + rng.normal(0, noise, H) for _ in layers]) for k in REPS}
            for k in REPS:
                arrs[k].append(vecs[k])
            norms.append(np.linalg.norm(np.stack([vecs[k] for k in REPS]), axis=-1))
            has_answer = bool(rng.random() > 0.02)
            d = r.to_dict()
            rows.append({**d, "config_id": "synthetic", "has_answer": has_answer, "truncated": not has_answer,
                         "finish_reason": "stop" if has_answer else "length", "answer_text": "x"})
            idx.append({"id": r.id, "row": i, "config_id": "synthetic", **{k: d.get(k) for k in META_FIELDS},
                        "n_input_tokens": 200, "n_reasoning_tokens": 100, "n_answer_tokens": 50 if has_answer else 0,
                        "think_closed": has_answer, "has_answer": has_answer, "n_generated": 160,
                        "truncated": not has_answer, "finish_reason": "stop" if has_answer else "length"})
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
    return {"shared": shared, "extra": extra}
