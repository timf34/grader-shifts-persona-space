"""Reproduce Betley et al.'s automated-vs-human grading vector (public-steering-vectors 0007)
with OUR extractor, and use it to verify our layer/position conventions.

Their recipe (vectors/0007/meta.json): Qwen/Qwen3.6-27B bf16, apply_chat_template(
add_generation_prompt=True, enable_thinking=True), residual at the INPUT of each decoder
block at the last prompt token, v_L = mean(positive) - mean(negative), layer 36.

Verification on Qwen3.6-27B (where their arrays exist):
  * per-layer cosine between our delta and theirs (should be ~1 at every layer if the
    conventions match; an off-by-one shows up as higher cosine at a shifted index)
  * per-layer mean activation norm vs theirs (norms change a lot between adjacent
    layers -- e.g. 124 at 54 vs 228 at 55 -- so the best-matching offset is unambiguous)
  * the first templated positive prompt must equal theirs byte for byte
For other models the vector is reproduced with the same recipe (no published reference).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import DATA_DIR
from .templates import prompt_token_ids, render_prompt

BETLEY_DIR = DATA_DIR / "betley_0007"


def load_prompt_file(path: Path) -> list[list[dict[str, str]]]:
    out = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "messages" in row:
                out.append(row["messages"])
            else:
                out.append([{"role": "user", "content": row["text"]}])
    return out


def published() -> dict[str, Any]:
    meta = json.loads((BETLEY_DIR / "meta.json").read_text())
    return {
        "meta": meta,
        "vector": np.load(BETLEY_DIR / "vector.npy"),
        "deltas": np.load(BETLEY_DIR / "deltas_all_layers.npy"),
        "act_norms": np.array(meta["per_layer_mean_activation_norm"], dtype=np.float64),
    }


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def reproduce(model, tokenizer, out_dir: Path, enable_thinking: bool = True,
              template_kwargs: dict | None = None) -> dict[str, Any]:
    """template_kwargs: the model's experiment template settings (e.g. Qwen3.8 reasoning_effort),
    so the reproduced direction lives in the same prompt format as the experiment."""
    from .extract import run_prompt_capture

    out_dir.mkdir(parents=True, exist_ok=True)
    pos = load_prompt_file(BETLEY_DIR / "positive.jsonl")
    neg = load_prompt_file(BETLEY_DIR / "negative.jsonl")
    pos_ids = [prompt_token_ids(tokenizer, m, enable_thinking, template_kwargs) for m in pos]
    neg_ids = [prompt_token_ids(tokenizer, m, enable_thinking, template_kwargs) for m in neg]
    a_pos, layers = run_prompt_capture(model, tokenizer, pos_ids, all_layers=True)
    a_neg, _ = run_prompt_capture(model, tokenizer, neg_ids, all_layers=True)
    deltas = a_pos.astype(np.float64).mean(0) - a_neg.astype(np.float64).mean(0)
    allacts = np.concatenate([a_pos, a_neg]).astype(np.float64)
    act_norms = np.linalg.norm(allacts, axis=-1).mean(0)
    np.save(out_dir / "deltas_all_layers.npy", deltas.astype(np.float32))
    np.save(out_dir / "act_norms.npy", act_norms.astype(np.float32))
    report: dict[str, Any] = {
        "n_pos": len(pos), "n_neg": len(neg), "layers": layers,
        "templated_first_positive_prompt": render_prompt(tokenizer, pos[0], enable_thinking, template_kwargs),
        "template_kwargs": template_kwargs or {},
        "prompt_tokens_mean_pos": float(np.mean([len(x) for x in pos_ids])),
        "per_layer_delta_norm": np.linalg.norm(deltas, axis=-1).tolist(),
        "per_layer_mean_activation_norm": act_norms.tolist(),
    }
    report.update(compare_with_published(deltas, act_norms, report["templated_first_positive_prompt"]))
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    return report


def compare_with_published(deltas: np.ndarray, act_norms: np.ndarray, templated_first: str | None = None,
                           primary_layer: int = 36) -> dict[str, Any]:
    pub = published()
    theirs, tnorm = pub["deltas"].astype(np.float64), pub["act_norms"]
    if theirs.shape != deltas.shape:
        return {"comparison": f"skipped: shape {deltas.shape} vs published {theirs.shape} (different model)"}
    n = deltas.shape[0]
    cos_same = [_cos(deltas[L], theirs[L]) for L in range(n)]
    offsets = {}
    for off in (-2, -1, 0, 1, 2):
        idx = [L for L in range(1, n) if 0 <= L + off < n]
        rel = [abs(act_norms[L] - tnorm[L + off]) / tnorm[L + off] for L in idx]
        cs = [_cos(deltas[L], theirs[L + off]) for L in idx if L + off >= 1]
        offsets[off] = {"median_rel_norm_err": float(np.median(rel)), "mean_cos": float(np.nanmean(cs))}
    best = min(offsets, key=lambda o: offsets[o]["median_rel_norm_err"])
    res = {
        "cos_with_published_by_layer": cos_same,
        "cos_with_published_at_primary": cos_same[primary_layer],
        "norm_ratio_at_primary": float(np.linalg.norm(deltas[primary_layer]) / np.linalg.norm(theirs[primary_layer])),
        "act_norm_rel_err_by_layer": [float(abs(a - b) / b) if b > 0 else None for a, b in zip(act_norms, tnorm)],
        "offset_scan": offsets,
        "best_offset_by_norm_profile": best,
        "templated_prompt_matches_published": (
            None if templated_first is None else templated_first == pub["meta"]["templated_first_positive_prompt"]),
    }
    res["verification_passed"] = bool(
        best == 0 and cos_same[primary_layer] > 0.95 and res["templated_prompt_matches_published"] is not False)
    return res
