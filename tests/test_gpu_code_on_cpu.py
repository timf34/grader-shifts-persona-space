"""The GPU-side code (templates, hooks, extraction, Betley comparison) on CPU.

Uses a tiny random Qwen3.5-architecture model (same classes as Qwen3.6/3.8-27B) and, when
cached locally, the real Qwen tokenizers. Tests needing a tokenizer are skipped otherwise
(populate the cache once with `python scripts/fetch_tokenizers.py`).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from gshift.config import get_model  # noqa: E402

REVS = {k: get_model(k) for k in ("qwen3.6-27b", "qwen3.8-27b")}


def tokenizer(key):
    spec = REVS[key]
    try:
        return transformers.AutoTokenizer.from_pretrained(spec.hf_id, revision=spec.revision, local_files_only=True)
    except Exception:
        pytest.skip(f"{spec.hf_id} tokenizer not cached (run scripts/fetch_tokenizers.py)")


def tiny_model(vocab: int = 248320):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    torch.manual_seed(0)
    tc = dict(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=8, num_attention_heads=4,
              num_key_value_heads=2, head_dim=8, linear_num_key_heads=2, linear_num_value_heads=4,
              linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4, full_attention_interval=4)
    cfg = Qwen3_5Config(text_config=tc, vision_config=dict(depth=1, hidden_size=16, intermediate_size=32,
                                                           num_heads=2, out_hidden_size=32))
    return Qwen3_5ForConditionalGeneration(cfg).eval()


def test_hook_convention_matches_hidden_states():
    from gshift.hooks import add_to_residual, capture_residual, find_text_model

    m = tiny_model(vocab=512)
    tm = find_text_model(m)
    ids = torch.randint(0, 512, (1, 12))
    with torch.no_grad():
        hs = tm(input_ids=ids, output_hidden_states=True).hidden_states
        with capture_residual(m, [0, 3, 7]) as st:
            tm(input_ids=ids)
    for L in (0, 3, 7):        # layer L == input of block L == hidden_states[L]
        assert torch.allclose(st[L], hs[L], atol=1e-6)
    v = torch.randn(32)
    with torch.no_grad(), capture_residual(m, [5]) as st2, add_to_residual(m, 5, v):
        tm(input_ids=ids)
    assert torch.allclose(st2[5], hs[5] + v, atol=1e-5)   # steering at layer 5 == input of block 5


@pytest.mark.parametrize("key", ["qwen3.6-27b", "qwen3.8-27b"])
def test_templates(key):
    from gshift.betley import BETLEY_DIR, load_prompt_file, published
    from gshift.templates import inspect_template, render_prompt, split_generation

    tok = tokenizer(key)
    kw = REVS[key].chat_template_kwargs
    info = inspect_template(tok, True, kw)
    assert info.prompt_opens_think and info.think_close_id is not None
    assert not info.notes, info.notes            # no injected system text with our settings
    pos = load_prompt_file(BETLEY_DIR / "positive.jsonl")
    if key == "qwen3.6-27b":
        assert render_prompt(tok, pos[0], True, kw) == published()["meta"]["templated_first_positive_prompt"]
    p = render_prompt(tok, [{"role": "user", "content": "Hi"}], True, kw)
    assert p == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    gen = tok("Think.\n</think>\n\nAnswer text.", add_special_tokens=False)["input_ids"] + [info.end_ids[-1]]
    sp = split_generation(gen, tok, info)
    assert sp.has_answer and tok.decode([gen[i] for i in sp.answer]).strip() == "Answer text."
    assert tok.decode([gen[i] for i in sp.reasoning]).strip() == "Think."


def test_end_to_end_extraction_matches_hidden_states(tmp_path, monkeypatch):
    """Fake generations -> extract_role -> Store; saved summaries equal HF hidden states."""
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.data import Store
    from gshift.design import build_requests
    from gshift.extract import Extractor, extract_role
    from gshift.hooks import find_text_model
    from gshift.io import append_jsonl, responses_path
    from gshift.templates import inspect_template, prompt_token_ids, split_generation

    tok = tokenizer("qwen3.6-27b")
    m = tiny_model()
    from gshift.config import load_experiment
    from gshift.smoke import smoke_experiment

    exp = smoke_experiment(load_experiment())
    reqs = [r for r in build_requests(exp, "main") if r.role == "librarian" and r.prompt_variant == 0][:6]
    info = inspect_template(tok, True)
    rng = np.random.default_rng(0)
    rows = []
    for i, r in enumerate(reqs):
        p_ids = prompt_token_ids(tok, r.messages, True)
        body = tok(f"reasoning {i}\n</think>\n\nfinal answer number {i}.", add_special_tokens=False)["input_ids"]
        if i == 5:   # truncated mid-reasoning: no answer
            body = tok("still thinking", add_special_tokens=False)["input_ids"]
        rows.append({**r.to_dict(), "config_id": "test", "prompt_ids": p_ids, "gen_ids": body + ([248046] if i < 5 else []),
                     "truncated": i == 5, "finish_reason": "length" if i == 5 else "stop"})
    for r in rows:
        p = responses_path("tiny", r["condition"], r["role"])
        p.parent.mkdir(parents=True, exist_ok=True)
        append_jsonl(p, [r])
    ex = Extractor(m, tok, [2, 5, 8])
    for cond in {r["condition"] for r in rows}:
        extract_role(ex, "tiny", cond, "librarian", save_dtype="float32")
    st = Store.load("tiny")
    assert len(st.meta) == 6 and st.layers == [2, 5, 8]
    assert st.meta.set_index("id").loc[rows[5]["id"], "has_answer"] == False  # noqa: E712
    tm = find_text_model(m)
    for r in rows[:5]:
        sub = st.meta[st.meta["id"] == r["id"]]
        P = len(r["prompt_ids"])
        sp = split_generation(r["gen_ids"], tok, info)
        with torch.no_grad():
            hs = tm(input_ids=torch.tensor([r["prompt_ids"] + r["gen_ids"]]), output_hidden_states=True).hidden_states
        for L in (2, 5, 8):
            pre = st.rows("pre", L, sub)[0]
            resp = st.rows("resp", L, sub)[0]
            want_pre = hs[L][0, P - 1].numpy() if L < 8 else None
            if want_pre is not None:
                assert np.allclose(pre, want_pre, atol=1e-4)
                want_resp = hs[L][0, [P + i for i in sp.answer]].double().mean(0).numpy()
                assert np.allclose(resp, want_resp, atol=1e-4)
    # answer tokens exclude </think>, the "\n\n" after it and <|im_end|>
    assert tok.decode([rows[0]["gen_ids"][i] for i in split_generation(rows[0]["gen_ids"], tok, info).answer]) \
        == "final answer number 0."


def test_betley_comparison_detects_offsets():
    from gshift.betley import compare_with_published, published

    pub = published()
    d, n = pub["deltas"].astype(np.float64), pub["act_norms"]
    ok = compare_with_published(d, n)
    assert ok["verification_passed"] and ok["best_offset_by_norm_profile"] == 0
    # "assistant-axis convention" (block OUTPUT) == our layer + 1: must be detected
    shifted = compare_with_published(np.roll(d, -1, axis=0), np.roll(n, -1))
    assert not shifted["verification_passed"] and shifted["best_offset_by_norm_profile"] == 1
