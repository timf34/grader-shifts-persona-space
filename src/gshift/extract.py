"""Stage 2 (assistant-axis pipeline/2_activations.py, adapted): residual-stream summaries
from a teacher-forced HF forward pass over the EXACT generated tokens.

Input per record: prompt_ids + gen_ids as stored by stage 1 -- no re-templating (Qwen
templates rewrite assistant turns, e.g. reasoning handling, so re-templating would not
reproduce what the model saw). Batch size 1: no padding, so no recurrent Gated-DeltaNet
state ever sees a pad token.

Saved per record, for each layer of the pre-registered sweep (convention in hooks.py):
  pre     residual at the final prompt position (the "\\n" after <think>) -- Betley et
          al.'s extraction position
  resp    mean over FINAL-ANSWER tokens (after </think>; special and boundary-whitespace
          tokens excluded), averaged within the response
  reason  mean over reasoning tokens (separate, optional representation -- never mixed
          with resp)
Records with no final answer (truncated mid-reasoning) get resp = 0 and has_answer=False;
the analysis excludes them by matched-pair rules.

Layout: outputs/<model>/<condition>/activations/<role>/{pre,resp,reason}.npy
        [n, n_layers_sweep, hidden] (float16 by default), norms.npy [n, 3, n_layers]
        (mean per-token L2 norm), index.jsonl (id, row, spans, counts), meta.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .config import ModelSpec
from .io import activations_dir, environment_info, read_jsonl, responses_path, write_json
from .templates import inspect_template, split_generation

REPS = ("pre", "resp", "reason")
# request metadata copied into index.jsonl so analysis never has to parse the (large)
# response files
META_FIELDS = ("role", "category", "role_split", "prompt_variant", "question_id", "question_group",
               "question_domain", "question_family", "question_split", "condition", "evaluator", "criterion",
               "wording_id", "wording_set", "baseline_id", "enable_thinking", "sample", "stage")


def text_config(config):
    return getattr(config, "text_config", None) or config


def load_hf_model(spec: ModelSpec, dtype: str = "bfloat16", device_map: str | dict = "auto"):
    """Load the decoder with all weights present; refuse a partially initialised model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    common = dict(revision=spec.revision, dtype=torch_dtype, device_map=device_map,
                  trust_remote_code=spec.trust_remote_code)
    model, info = AutoModelForCausalLM.from_pretrained(spec.hf_id, output_loading_info=True, **common)
    missing = [k for k in sorted(info.get("missing_keys", [])) if "layers." in k or "embed_tokens" in k]
    if missing:
        print(f"[extract] AutoModelForCausalLM left {len(missing)} decoder weights uninitialised "
              f"(e.g. {missing[:3]}); falling back to AutoModelForImageTextToText", flush=True)
        from transformers import AutoModelForImageTextToText

        del model
        model, info = AutoModelForImageTextToText.from_pretrained(spec.hf_id, output_loading_info=True, **common)
        missing = [k for k in sorted(info.get("missing_keys", [])) if "layers." in k or "embed_tokens" in k]
        if missing:
            raise RuntimeError(f"decoder weights missing after fallback: {missing[:5]}")
    model.eval()
    tok = AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.tokenizer_revision or spec.revision,
                                        trust_remote_code=spec.trust_remote_code)
    tc = text_config(model.config)
    if spec.n_layers is not None and tc.num_hidden_layers != spec.n_layers:
        raise RuntimeError(f"{spec.hf_id}: {tc.num_hidden_layers} layers, configs/models.yaml says {spec.n_layers}")
    if spec.hidden_size is not None and tc.hidden_size != spec.hidden_size:
        raise RuntimeError(f"{spec.hf_id}: hidden {tc.hidden_size}, configs/models.yaml says {spec.hidden_size}")
    return model, tok


def input_device(model):
    return model.get_input_embeddings().weight.device


class Extractor:
    def __init__(self, model, tokenizer, layers: list[int], enable_thinking: bool = True,
                 template_kwargs: dict | None = None):
        from .hooks import find_text_model

        self.model = model
        # Run the text decoder directly: no LM head, no multimodal wrapper. For text-only
        # input Qwen3.5-style M-RoPE positions equal plain arange positions, which is what
        # the text model computes by default.
        self.text_model = find_text_model(model)
        self.tok = tokenizer
        self.n_layers = len(self.text_model.layers)
        self.layers = sorted(set(int(x) for x in layers))
        for L in self.layers:
            if not 0 <= L <= self.n_layers:
                raise ValueError(f"layer {L} out of range for {self.n_layers}-block model")
        self.info = inspect_template(tokenizer, enable_thinking, template_kwargs)

    def forward_capture(self, ids: list[int]) -> dict[int, Any]:
        import torch

        from .hooks import capture_residual

        x = torch.tensor([ids], dtype=torch.long, device=input_device(self.model))
        with torch.inference_mode(), capture_residual(self.model, self.layers, self.n_layers) as store:
            self.text_model(input_ids=x, use_cache=False)
        return {L: store[L][0] for L in self.layers}  # [T, H] each

    def record(self, rec: dict[str, Any]) -> dict[str, Any]:
        import torch

        p_ids, g_ids = rec["prompt_ids"], rec["gen_ids"]
        split = split_generation(g_ids, self.tok, self.info)
        P = len(p_ids)
        last_needed = max([-1] + split.answer + split.reasoning)
        ids = list(p_ids) + list(g_ids[: last_needed + 1])
        hs = self.forward_capture(ids)
        H = next(iter(hs.values())).shape[-1]
        nL = len(self.layers)
        out = {rep: np.zeros((nL, H), dtype=np.float32) for rep in REPS}
        norms = np.zeros((len(REPS), nL), dtype=np.float32)
        a_idx = torch.tensor([P + i for i in split.answer], dtype=torch.long)
        r_idx = torch.tensor([P + i for i in split.reasoning], dtype=torch.long)
        for j, L in enumerate(self.layers):
            h = hs[L]
            pre = h[P - 1].double()
            out["pre"][j] = pre.cpu().numpy()
            norms[0, j] = float(pre.norm())
            for k, (rep, idx) in enumerate((("resp", a_idx), ("reason", r_idx)), start=1):
                if len(idx):
                    sel = h[idx.to(h.device)].double()
                    out[rep][j] = sel.mean(0).cpu().numpy()
                    norms[k, j] = float(sel.norm(dim=-1).mean())
        return {"acts": out, "norms": norms, "n_input_tokens": len(ids), "split": split}

    def prompt_last(self, ids: list[int]) -> np.ndarray:
        """[n_layers_captured, H] residual at the last position of `ids`."""
        hs = self.forward_capture(ids)
        return np.stack([hs[L][-1].double().cpu().numpy() for L in self.layers]).astype(np.float32)


def _role_done(out_dir: Path, ids: list[str], config_ids: set[str]) -> bool:
    idx = out_dir / "index.jsonl"
    if not idx.exists() or not (out_dir / "pre.npy").exists():
        return False
    rows = read_jsonl(idx)
    return [r["id"] for r in rows] == ids and {r.get("config_id") for r in rows} <= config_ids


def extract_role(ex: Extractor, model_key: str, condition: str, role: str, save_dtype: str = "float16",
                 extra_meta: dict | None = None) -> int:
    """Extract one (condition, role) file. Re-extracts the role if its responses changed."""
    recs = read_jsonl(responses_path(model_key, condition, role))
    if not recs:
        return 0
    out_dir = activations_dir(model_key, condition, role)
    ids = [r["id"] for r in recs]
    if _role_done(out_dir, ids, {r.get("config_id") for r in recs}):
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {rep: [] for rep in REPS}
    norms, index = [], []
    for i, rec in enumerate(recs):
        res = ex.record(rec)
        for rep in REPS:
            arrays[rep].append(res["acts"][rep])
        norms.append(res["norms"])
        sp = res["split"]
        index.append({"id": rec["id"], "row": i, "config_id": rec.get("config_id"),
                      **{k: rec.get(k) for k in META_FIELDS},
                      "n_input_tokens": res["n_input_tokens"], **sp.to_dict(),
                      "answer_span": [sp.answer[0], sp.answer[-1] + 1] if sp.answer else None,
                      "reasoning_span": [sp.reasoning[0], sp.reasoning[-1] + 1] if sp.reasoning else None,
                      "truncated": rec.get("truncated"), "finish_reason": rec.get("finish_reason")})
    dt = np.dtype(save_dtype)
    for rep in REPS:   # arrays first, index last: an index file implies complete arrays
        np.save(out_dir / f"{rep}.npy", np.stack(arrays[rep]).astype(dt))
    np.save(out_dir / "norms.npy", np.stack(norms).astype(np.float32))
    write_json(out_dir / "meta.json", {"layers": ex.layers, "reps": list(REPS), "save_dtype": save_dtype,
                                       "n_layers_model": ex.n_layers, **(extra_meta or {})})
    tmp = out_dir / "index.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r) + "\n" for r in index))
    tmp.replace(out_dir / "index.jsonl")
    return len(recs)


def run_extraction(model, tokenizer, model_key: str, pairs: list[tuple[str, str]], layers: list[int],
                   enable_thinking: bool = True, template_kwargs: dict | None = None,
                   save_dtype: str = "float16", extra_meta: dict | None = None) -> None:
    ex = Extractor(model, tokenizer, layers, enable_thinking, template_kwargs)
    meta = {"layers": ex.layers,
            "layer_convention": "layer L = residual stream at the INPUT of decoder block L (HF hidden_states[L]); "
                                "L == n_layers = output of the last block",
            "template": ex.info.to_dict(), "environment": environment_info(), **(extra_meta or {})}
    t0, n = time.time(), 0
    for k, (cond, role) in enumerate(pairs):
        m = extract_role(ex, model_key, cond, role, save_dtype, meta)
        n += m
        if m:
            print(f"[extract] {cond}/{role}: {m} records ({k + 1}/{len(pairs)} files, "
                  f"{(time.time() - t0) / max(n, 1):.2f}s/record)", flush=True)
    print(f"[extract] done: {n} records in {time.time() - t0:.0f}s", flush=True)


def run_prompt_capture(model, tokenizer, prompts: list[list[int]], all_layers: bool = True,
                       layers: list[int] | None = None) -> tuple[np.ndarray, list[int]]:
    """Last-prompt-position residuals for token-id prompts -> ([N, nL, H], layers)."""
    from .hooks import find_decoder_layers

    n = len(find_decoder_layers(model))
    layers = list(range(n)) if all_layers else list(layers or [])
    ex = Extractor(model, tokenizer, layers)
    acts = np.stack([ex.prompt_last(ids) for ids in prompts])
    return acts, ex.layers
