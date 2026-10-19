"""Stage 1 (assistant-axis pipeline/1_generate.py, adapted): vLLM batched generation with
thinking enabled, storing exact token ids.

Changes vs the original pipeline (plan section "Implementation", A-D):
  * prompts are rendered once with the HF tokenizer (thinking ON, model-specific template
    kwargs) and passed to vLLM as token ids, so the stored prompt_ids are exactly what the
    model conditioned on; the extractor re-feeds prompt_ids + gen_ids verbatim;
  * every record carries role / prompt variant / question id / condition / seed and a
    config id (hash of model revision, template settings, sampling, design + manifest
    version). Resume skips a request only if a record with the same id AND config id
    exists; a record with the same id but another config id aborts the run, so results
    from different settings are never mixed or mistaken for each other;
  * outputs go to outputs/<model>/<condition>/responses/<role>.jsonl.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .config import ModelSpec, manifest_sha
from .design import Request
from .io import append_jsonl, environment_info, model_dir, read_jsonl, responses_path, write_json
from .templates import answer_text, inspect_template, prompt_token_ids, reasoning_text, render_prompt, split_generation

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
                 "max_tokens", "max_model_len", "enable_thinking", "samples_per_prompt", "base_seed")


def config_identity(spec: ModelSpec, exp: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    gen = exp["generation"]
    ident = {
        "model": spec.hf_id, "revision": spec.revision,
        "tokenizer_revision": spec.tokenizer_revision or spec.revision,
        "chat_template_kwargs": spec.chat_template_kwargs,
        "generation": {k: gen.get(k) for k in SAMPLING_KEYS},
        "design_version": exp["design_version"], "placement": exp.get("placement"),
        "question_manifest": exp["question_manifest"], "question_manifest_sha": manifest_sha(exp),
    }
    cid = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16]
    return cid, ident


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.tokenizer_revision or spec.revision,
                                         trust_remote_code=spec.trust_remote_code)


def build_llm(spec: ModelSpec, gen_cfg: dict[str, Any], seed: int = 0):
    from vllm import LLM

    kwargs = dict(
        model=spec.hf_id, revision=spec.revision,
        tokenizer_revision=spec.tokenizer_revision or spec.revision,
        tensor_parallel_size=spec.tensor_parallel_size,
        gpu_memory_utilization=spec.gpu_memory_utilization,
        max_model_len=int(gen_cfg["max_model_len"]), max_num_seqs=spec.max_num_seqs,
        dtype="bfloat16", seed=seed, trust_remote_code=spec.trust_remote_code,
    )
    kwargs.update(spec.vllm_kwargs or {})
    return LLM(**kwargs)


def sampling_params(gen_cfg: dict[str, Any], seed: int, max_tokens: int):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=float(gen_cfg["temperature"]), top_p=float(gen_cfg["top_p"]), top_k=int(gen_cfg["top_k"]),
        min_p=float(gen_cfg.get("min_p", 0.0)), presence_penalty=float(gen_cfg.get("presence_penalty", 0.0)),
        repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)), max_tokens=int(max_tokens),
        seed=int(seed), skip_special_tokens=False,
    )


def existing_records(model_key: str, requests: list[Request]) -> dict[str, str]:
    """id -> config_id of records already on disk for the (condition, role) files touched."""
    out: dict[str, str] = {}
    for cond, role in sorted({(r.condition, r.role) for r in requests}):
        for rec in read_jsonl(responses_path(model_key, cond, role)):
            out[rec["id"]] = rec.get("config_id", "?")
    return out


def run_generation(spec: ModelSpec, requests: list[Request], exp: dict[str, Any], chunk_size: int = 2048,
                   llm=None, tokenizer=None, extra_meta: dict[str, Any] | None = None) -> int:
    """Generate every request not yet on disk under the current config. Returns #generated."""
    gen_cfg = exp["generation"]
    cid, ident = config_identity(spec, exp)
    have = existing_records(spec.key, requests)
    clash = [i for i, c in have.items() if c != cid and i in {r.id for r in requests}]
    if clash:
        raise RuntimeError(
            f"{len(clash)} existing records (e.g. {clash[0]}) were generated under config id "
            f"{have[clash[0]]}, current is {cid}. Refusing to mix configurations: move the old outputs "
            "away or point GSHIFT_OUTPUTS elsewhere.")
    todo = [r for r in requests if r.id not in have]
    print(f"[generate] {spec.key} config={cid}: {len(requests)} requests, {len(requests) - len(todo)} done, "
          f"{len(todo)} to go", flush=True)
    tokenizer = tokenizer or load_tokenizer(spec)
    enable_thinking = bool(gen_cfg.get("enable_thinking", True))
    tkw = spec.chat_template_kwargs
    info = inspect_template(tokenizer, enable_thinking, tkw)
    if info.notes:
        print("[generate] template notes: " + "; ".join(info.notes), flush=True)
    write_json(model_dir(spec.key) / "run_config" / f"{cid}.json", {
        "config_id": cid, "identity": ident, "template": info.to_dict(), "environment": environment_info(),
        **(extra_meta or {})})
    if not todo:
        return 0
    if enable_thinking and info.think_close_id is None:
        raise RuntimeError("thinking enabled but tokenizer has no single </think> token")

    from vllm.inputs import TokensPrompt

    llm = llm or build_llm(spec, gen_cfg, seed=int(gen_cfg["base_seed"]))
    max_len = int(gen_cfg["max_model_len"])
    n_done = 0
    for start in range(0, len(todo), chunk_size):
        chunk = todo[start:start + chunk_size]
        prompts, params, pids = [], [], []
        for r in chunk:
            ids = prompt_token_ids(tokenizer, r.messages, enable_thinking, tkw)
            budget = min(int(gen_cfg["max_tokens"]), max_len - len(ids))
            if budget < 16:
                raise RuntimeError(f"prompt for {r.id} too long ({len(ids)} tokens)")
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            params.append(sampling_params(gen_cfg, r.seed, budget))
            pids.append(ids)
        t0 = time.time()
        outs = llm.generate(prompts, params, use_tqdm=True)
        by_file: dict[tuple[str, str], list[dict]] = defaultdict(list)
        n_tok = 0
        for r, ids, out, sp in zip(chunk, pids, outs, params):
            o = out.outputs[0]
            gen_ids = list(o.token_ids)
            n_tok += len(gen_ids)
            spl = split_generation(gen_ids, tokenizer, info)
            row = r.to_dict()
            row.update(
                model_key=spec.key, config_id=cid,
                prompt_text=render_prompt(tokenizer, r.messages, enable_thinking, tkw),
                prompt_ids=ids, prompt_sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16],
                gen_ids=gen_ids, gen_text=o.text, finish_reason=o.finish_reason,
                truncated=o.finish_reason == "length", max_tokens=sp.max_tokens,
                answer_span=[spl.answer[0], spl.answer[-1] + 1] if spl.answer else None,
                reasoning_span=[spl.reasoning[0], spl.reasoning[-1] + 1] if spl.reasoning else None,
                answer_text=answer_text(gen_ids, tokenizer, spl),
                reasoning_text=reasoning_text(gen_ids, tokenizer, spl),
                **spl.to_dict(),
            )
            by_file[(r.condition, r.role)].append(row)
        for (cond, role), rows in by_file.items():
            p = responses_path(spec.key, cond, role)
            p.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(p, rows)
        n_done += len(chunk)
        rows_all = [x for v in by_file.values() for x in v]
        dt = time.time() - t0
        print(f"[generate] {n_done}/{len(todo)}  {dt:.0f}s  {n_tok / max(dt, 1e-9):.0f} tok/s  "
              f"truncated={sum(x['truncated'] for x in rows_all)}  "
              f"has_answer={sum(x['has_answer'] for x in rows_all)}/{len(rows_all)}", flush=True)
    return n_done
