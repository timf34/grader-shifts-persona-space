"""EasySteer generation under named steering conditions -- client side.

Follows the setup verified in timf34/AttractorStatePrefillAttack (capped/easysteer): an
EasySteer vLLM server (vllm==0.26.0 + ZJU-REAL/EasySteer-vllm-v1@6267ca0 overlay, started
by scripts/pod/serve_easysteer.sh with --enable-steer-vector --steer-algorithms direct)
and one OpenAI-compatible request per prompt carrying its SteeringSpec in
`extra_body["steering"]`. The field is OMITTED for unsteered requests (the 0.26 overlay
rejects `false`). `direct` payloads must be GGUF files with tensors named `direction.<L>`.

Location: EasySteer layer k = OUTPUT of decoder block k (its hook adds the delta to the
complete stream hidden + residual), so our layer L (input of block L) is EasySteer layer
L-1 -- the same tensor as public-steering-vectors' `block = layer - 1`.

Prompts are sent as TOKEN IDS rendered by the HF tokenizer with the model's template
kwargs (thinking on, Qwen3.8 reasoning_effort), so steered and Q1 prompts are identical.

Condition names:  none | natural_auto | natural_human | <direction>_<+|-><strength>
applied vector = sign x strength x A_L x unit(direction).
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ModelSpec
from ..io import append_jsonl, read_jsonl
from ..templates import answer_text, inspect_template, prompt_token_ids, reasoning_text, split_generation
from .tasks import Prompt

_COND = re.compile(r"^(?P<dir>.+)_(?P<sign>[+-])(?P<s>[0-9.]+)$")


def parse_condition(name: str) -> tuple[str | None, float]:
    """-> (direction or None, signed strength)."""
    if name in ("none", "natural_auto", "natural_human"):
        return None, 0.0
    m = _COND.match(name)
    if not m:
        raise ValueError(f"bad steering condition {name!r}")
    s = float(m["s"])
    return m["dir"], s if m["sign"] == "+" else -s


def condition_name(direction: str, signed_strength: float) -> str:
    return f"{direction}_{'+' if signed_strength >= 0 else '-'}{abs(signed_strength):g}"


def write_direction_gguf(unit: np.ndarray, our_layer: int, path: Path) -> Path:
    """EasySteer direction GGUF: tensor `direction.<our_layer - 1>` = unit vector (float32)."""
    import gguf

    path.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(path), arch="steer")
    w.add_tensor(f"direction.{our_layer - 1}", np.asarray(unit, dtype=np.float32).reshape(-1))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def steering_spec(gguf_path: Path, our_layer: int, scale: float) -> dict[str, Any]:
    return {"vectors": [{"source": str(Path(gguf_path).resolve()), "algorithm": "direct", "scale": float(scale),
                         "layers": [our_layer - 1], "apply": {"prompt": "all", "generation": "all"}}],
            "conflict": "sequential"}


def _seed(base: int, pid: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{base}|{pid}".encode()).digest()[:4], "little") & 0x7FFFFFFF


class SteerClient:
    def __init__(self, base_url: str, served_name: str, concurrency: int = 128, timeout: float = 3600):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key="-", timeout=timeout, max_retries=3)
        self.model = served_name
        self.concurrency = concurrency

    async def _one(self, sem, ids: list[int], gen: dict, seed: int, max_tokens: int, steering: dict | None):
        body: dict[str, Any] = {"top_k": int(gen["top_k"]), "min_p": float(gen.get("min_p", 0.0)),
                                "skip_special_tokens": False, "return_token_ids": True}
        if steering:
            body["steering"] = steering
        async with sem:
            r = await self.client.completions.create(
                model=self.model, prompt=ids, max_tokens=max_tokens, temperature=float(gen["temperature"]),
                top_p=float(gen["top_p"]), presence_penalty=float(gen.get("presence_penalty", 0.0)), seed=seed,
                extra_body=body)
        c = r.choices[0]
        tok = getattr(c, "token_ids", None)
        if tok is None and getattr(c, "model_extra", None):
            tok = c.model_extra.get("token_ids")
        return c.text, (list(tok) if tok is not None else None), c.finish_reason

    def run(self, jobs: list[tuple[list[int], dict, int, int, dict | None]]):
        async def go():
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(*(self._one(sem, *j) for j in jobs))

        return asyncio.run(go())


def generate_condition(client: SteerClient, tokenizer, spec: ModelSpec, prompts: list[Prompt], condition: str,
                       gguf_paths: dict[str, Path], manifest: dict[str, Any], gen_cfg: dict[str, Any],
                       out_path: Path) -> int:
    """Generate all prompts not yet in out_path under one condition (same seeds across conditions)."""
    done = {r["id"] for r in read_jsonl(out_path)}
    todo = [p for p in prompts if p.id not in done]
    if not todo:
        return 0
    direction, signed = parse_condition(condition)
    L = int(manifest["layer"])
    scale = signed * float(manifest["activation_norm"])
    steering = None if direction is None else steering_spec(gguf_paths[direction], L, scale)
    info = inspect_template(tokenizer, True, spec.chat_template_kwargs)
    ids = [prompt_token_ids(tokenizer, p.messages, True, spec.chat_template_kwargs) for p in todo]
    jobs = [(i, gen_cfg, _seed(int(gen_cfg["base_seed"]), p.id),
             min(int(gen_cfg["max_tokens"]), int(gen_cfg["max_model_len"]) - len(i)), steering)
            for p, i in zip(todo, ids)]
    t0 = time.time()
    outs = client.run(jobs)
    rows = []
    for p, i, (text, g, finish) in zip(todo, ids, outs):
        exact = g is not None
        if g is None:   # server did not return ids: re-tokenise the text (flagged)
            g = tokenizer(text, add_special_tokens=False)["input_ids"]
        sp = split_generation(g, tokenizer, info)
        rows.append({"id": p.id, "benchmark": p.benchmark, "condition": condition, "direction": direction,
                     "signed_strength": signed, "scale": scale, "steer_layer_ours": L,
                     "steer_layer_easysteer": L - 1 if direction else None, "messages": p.messages, "meta": p.meta,
                     "prompt_ids": i, "gen_ids": g, "gen_ids_exact": exact, "gen_text": text,
                     "finish_reason": finish, "truncated": finish == "length",
                     "answer_text": answer_text(g, tokenizer, sp), "reasoning_text": reasoning_text(g, tokenizer, sp),
                     **sp.to_dict()})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(out_path, rows)
    print(f"[steer] {condition} {prompts[0].benchmark}: {len(rows)} in {time.time() - t0:.0f}s; "
          f"has_answer={sum(r['has_answer'] for r in rows)}/{len(rows)}", flush=True)
    return len(rows)


def sanity_check(client: SteerClient, tokenizer, spec: ModelSpec, gguf_path: Path, layer: int,
                 act_norm: float) -> dict:
    """Greedy: scale 0 must equal unsteered, and a large scale must change the output."""
    msgs = [{"role": "user", "content": "Name three primary colours."}]
    ids = prompt_token_ids(tokenizer, msgs, True, spec.chat_template_kwargs)
    g = {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}
    base, zero, big = client.run([(ids, g, 0, 64, None), (ids, g, 0, 64, steering_spec(gguf_path, layer, 0.0)),
                                  (ids, g, 0, 64, steering_spec(gguf_path, layer, 4.0 * act_norm))])
    res = {"zero_scale_matches_unsteered": base[0] == zero[0], "large_scale_changes_output": base[0] != big[0],
           "unsteered_text": base[0], "large_scale_text": big[0], "token_ids_returned": base[1] is not None}
    res["ok"] = res["zero_scale_matches_unsteered"] and res["large_scale_changes_output"]
    return res
