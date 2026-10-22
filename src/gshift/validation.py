"""Checks that need real generations or a GPU, written so their logic is CPU-testable with
a tiny model (tests/test_gpu_code_on_cpu.py). Driven by scripts/gpu_checks.py; see
docs/gpu_validation.md.

  spans   token-span parsing on real generations: re-split every record and check the
          reasoning / final-answer invariants of the model's actual output format
  replay  exact-token replay: teacher-forcing prompt_ids + gen_ids through the HF model must
          rank every vLLM-sampled token inside the sampler's top-k (an off-by-one or
          re-templated replay fails this immediately; a shifted-by-one control shows the
          check has power)
  steer   (a) zero-strength EasySteer output == unsteered output (greedy and seeded
          sampling); (b) a nonzero intervention lands at the intended layer, positions, sign
          and scale: HF with strength x A_L x unit(v) added at OUR layer L reproduces the
          server's per-token log-probs better than layer L-1 / L+1, scale x0.5 / x2 or the
          opposite sign (EasySteer layer = L - 1), and the GGUF payload round-trips
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np

from .templates import THINK_CLOSE, THINK_OPEN, TemplateInfo, answer_text, split_generation


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------

def check_spans(rows: list[dict], tokenizer, info: TemplateInfo, max_examples: int = 5) -> dict[str, Any]:
    """Invariants of stored generations (one generation config, thinking per info.thinking)."""
    hard: dict[str, list[str]] = {}
    soft: dict[str, list[str]] = {}

    def flag(d, k, rid):
        d.setdefault(k, [])
        if len(d[k]) < max_examples:
            d[k].append(rid)
        d.setdefault(f"_n_{k}", [])
        d[f"_n_{k}"].append(rid)

    end = set(info.end_ids)
    special = set(info.special_ids)
    for r in rows:
        g, rid = list(r["gen_ids"]), r["id"]
        sp = split_generation(g, tokenizer, info)
        stored_a = r.get("answer_span")
        got_a = [sp.answer[0], sp.answer[-1] + 1] if sp.answer else None
        if stored_a != got_a or bool(r.get("has_answer")) != sp.has_answer:
            flag(hard, "stored_span_differs_from_resplit", rid)
        if info.think_close_id is None:     # multi-token tags (OLMo 3): count them in the decoded text
            txt = tokenizer.decode(g, skip_special_tokens=False)
            n_open, n_close = txt.count(THINK_OPEN), txt.count(THINK_CLOSE)
        else:
            n_open, n_close = g.count(info.think_open_id), g.count(info.think_close_id)
        if info.thinking:
            if info.prompt_opens_think and n_open:
                flag(soft, "model_emitted_extra_think_open", rid)
            if n_close > 1:
                flag(soft, "multiple_think_close", rid)
            if r.get("finish_reason") == "stop" and not sp.think_closed:
                flag(hard, "stopped_without_closing_think", rid)
        else:
            if n_open or n_close:
                flag(soft, "thinking_off_but_think_tags_generated", rid)
            if sp.reasoning:
                flag(hard, "thinking_off_but_reasoning_span", rid)
        if r.get("finish_reason") == "stop" and (not g or g[-1] not in end):
            flag(hard, "stop_without_end_token", rid)
        if r.get("truncated") and len(g) != int(r.get("max_tokens", len(g))):
            flag(hard, "truncated_but_not_at_budget", rid)
        if any(g[i] in special for i in sp.answer) or any(g[i] in special for i in sp.reasoning):
            flag(hard, "special_token_inside_span", rid)
        if sp.answer and answer_text(g, tokenizer, sp) != r.get("answer_text", answer_text(g, tokenizer, sp)):
            flag(hard, "answer_text_mismatch", rid)
        if sp.has_answer and r.get("truncated"):
            flag(soft, "truncated_after_answer_started", rid)
    counts = lambda d: {k[3:]: len(v) for k, v in d.items() if k.startswith("_n_")}  # noqa: E731
    return {"n_records": len(rows), "thinking": info.thinking,
            "hard_failures": counts(hard), "warnings": counts(soft),
            "examples": {k: v for k, v in {**hard, **soft}.items() if not k.startswith("_n_")},
            "think_closed_rate": float(np.mean([split_generation(r["gen_ids"], tokenizer, info).think_closed
                                                for r in rows])) if rows else None,
            "ok": not any(counts(hard).values())}


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def _gen_logits(model, prompt_ids: Sequence[int], gen_ids: Sequence[int]):
    """Float logits rows predicting gen[0..G-1] (teacher-forced), computing only the
    generated positions' logits when the model supports `logits_to_keep`."""
    import torch

    dev = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids) + list(gen_ids)], device=dev)
    keep = len(gen_ids) + 1
    with torch.no_grad():
        try:
            out = model(input_ids=ids, logits_to_keep=keep).logits[0]
        except TypeError:
            out = model(input_ids=ids).logits[0]
    return out[-keep:-1].float()


def token_ranks(logits, targets) -> np.ndarray:
    """Rank (0 = argmax) of each target token under the corresponding logits row."""
    import torch

    tl = logits.gather(-1, targets[:, None])
    return (logits > tl).sum(-1).cpu().numpy()


def replay_check(model, rows: list[dict], top_k: int, min_frac: float = 0.99, max_gen: int = 2048) -> dict[str, Any]:
    """Teacher-forced HF forward over prompt_ids + gen_ids (batch size 1, as extraction;
    the first `max_gen` generated tokens, which is still an exact prefix). Every token vLLM
    sampled with top-k sampling must sit inside HF's top-k at its position; the control
    re-aligns targets one position late."""
    import torch

    dev = next(model.parameters()).device
    in_k, ctrl_in_k, n_tok, lp = 0, 0, 0, []
    per = []
    for r in rows:
        p, g = list(r["prompt_ids"]), list(r["gen_ids"])[:max_gen]
        if len(g) < 2:
            continue
        pred = _gen_logits(model, p, g)                           # row t predicts gen[t]
        tgt = torch.tensor(g, device=dev)
        rk = token_ranks(pred, tgt)
        ctrl = token_ranks(pred[1:], tgt[:-1])                     # off by one: row t+1 vs gen[t]
        logp = torch.log_softmax(pred, -1).gather(-1, tgt[:, None])[:, 0].cpu().numpy()
        in_k += int((rk < top_k).sum())
        ctrl_in_k += int((ctrl < top_k).sum())
        n_tok += len(g)
        lp.append(float(logp.mean()))
        per.append({"id": r["id"], "n_gen": len(g), "frac_in_top_k": float((rk < top_k).mean()),
                    "max_rank": int(rk.max())})
    frac = in_k / max(n_tok, 1)
    ctrl_frac = ctrl_in_k / max(n_tok - len(per), 1)
    return {"n_records": len(per), "n_tokens": n_tok, "top_k": top_k, "frac_in_top_k": frac,
            "control_off_by_one_frac_in_top_k": ctrl_frac, "mean_token_logprob": float(np.mean(lp)) if lp else None,
            "worst_records": sorted(per, key=lambda x: x["frac_in_top_k"])[:5],
            "ok": bool(per) and frac >= min_frac and ctrl_frac < frac - 0.05}


# ---------------------------------------------------------------------------
# steering
# ---------------------------------------------------------------------------

def steered_token_logprobs(model, prompt_ids: Sequence[int], gen_ids: Sequence[int], layer: int | None,
                           vector=None) -> np.ndarray:
    """log p(gen[t] | prefix) under HF with `vector` added at OUR layer (input of block
    `layer`, every position), teacher-forced on the given tokens."""
    import torch

    from .hooks import add_to_residual

    dev = next(model.parameters()).device
    if layer is None:
        pred = _gen_logits(model, prompt_ids, gen_ids)
    else:
        with add_to_residual(model, layer, torch.as_tensor(vector)):
            pred = _gen_logits(model, prompt_ids, gen_ids)
    return torch.log_softmax(pred, -1).gather(-1, torch.tensor(list(gen_ids), device=dev)[:, None])[:, 0].cpu().numpy()


def steering_variants(layer: int, n_layers: int) -> list[tuple[str, int, float]]:
    """(name, our layer, scale multiplier); sign handled by the caller."""
    out = [("L", layer, 1.0), ("L_x0.5", layer, 0.5), ("L_x2", layer, 2.0)]
    if layer - 1 >= 1:
        out.append(("L-1", layer - 1, 1.0))
    if layer + 1 <= n_layers:
        out.append(("L+1", layer + 1, 1.0))
    return out


def compare_steering(model, cases: list[dict], unit: np.ndarray, layer: int, n_layers: int,
                     max_tokens: int = 24) -> dict[str, Any]:
    """cases: [{prompt_ids, gen_ids, token_logprobs, signed_scale}] from the server (greedy,
    steered with signed_scale x unit at our layer). For every case, the mean |Delta logprob|
    between the server and HF under each variant (incl. the opposite sign); the intended
    variant must be the closest in every steered case."""
    import torch

    res = []
    for c in cases:
        g, lp_srv = c["gen_ids"][:max_tokens], np.asarray(c["token_logprobs"][:max_tokens], float)
        if c["signed_scale"] == 0:
            hf = steered_token_logprobs(model, c["prompt_ids"], g, None)
            res.append({"signed_scale": 0.0, "mae": {"unsteered": float(np.abs(hf - lp_srv).mean())}})
            continue
        mae = {}
        for sign_name, sgn in (("", 1.0), ("_flipped_sign", -1.0)):
            for name, L, mult in steering_variants(layer, n_layers):
                if sign_name and name != "L":
                    continue
                v = torch.tensor(unit * c["signed_scale"] * mult * sgn, dtype=torch.float32)
                hf = steered_token_logprobs(model, c["prompt_ids"], g, L, v)
                mae[name + sign_name] = float(np.abs(hf - lp_srv).mean())
        hf0 = steered_token_logprobs(model, c["prompt_ids"], g, None)
        mae["unsteered"] = float(np.abs(hf0 - lp_srv).mean())
        best = min(mae, key=mae.get)
        res.append({"signed_scale": float(c["signed_scale"]), "mae": mae, "best": best})
    steered = [r for r in res if r["signed_scale"] != 0]
    return {"cases": res, "best_counts": dict(Counter(r["best"] for r in steered)),
            "ok": bool(steered) and all(r["best"] == "L" for r in steered)}


def gguf_roundtrip(path, unit: np.ndarray, our_layer: int) -> dict[str, Any]:
    """The EasySteer payload holds exactly unit(v) under tensor `direction.<L-1>`."""
    import gguf

    rd = gguf.GGUFReader(str(path))
    names = [t.name for t in rd.tensors]
    want = f"direction.{our_layer - 1}"
    t = next((t for t in rd.tensors if t.name == want), None)
    arr = None if t is None else np.asarray(t.data, dtype=np.float32).reshape(-1)
    ok = t is not None and arr.shape == unit.shape and np.allclose(arr, unit, atol=1e-6)
    return {"tensors": names, "expected_tensor": want, "ok": bool(ok),
            "max_abs_diff": float(np.abs(arr - unit).max()) if arr is not None and arr.shape == unit.shape else None}
