"""Q2 code paths on CPU (no vLLM / EasySteer needed)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gshift import CONFIG_DIR
from gshift.config import load_experiment, load_yaml
from gshift.steer import calibrate, metrics, tasks
from gshift.steer.engine import condition_name, parse_condition


@pytest.fixture
def cfg():
    return load_yaml(CONFIG_DIR / "steering.yaml")


def test_prompt_sets(cfg):
    exp = load_experiment()
    assert len(tasks.benchmark_prompts("sorh", cfg, exp)) == 306
    d = tasks.benchmark_prompts("sorh_descored", cfg, exp)
    assert len(d) == 306 and d[0].messages[0]["content"] != tasks.benchmark_prompts("sorh", cfg, exp)[0].messages[0]["content"]
    g = tasks.benchmark_prompts("gsm8k", cfg, exp)
    assert len(g) == 200 and g[0].messages[0]["content"].startswith("Solve the following problem.")
    per = tasks.benchmark_prompts("persona", cfg, exp)
    assert len(per) == 10 * 20                       # held-out roles x validation questions
    cal = tasks.calibration_prompts(cfg, exp)
    q1 = {q["id"] for q in __import__("gshift.config", fromlist=["x"]).load_questions(exp)}
    assert len(cal) == 24 + 20 and not any(p.id.replace("calib_", "orig_") in q1 for p in cal)
    nat = tasks.benchmark_prompts("sorh", cfg, exp, tasks.natural_prefix("natural_auto", exp))
    assert nat[0].messages[0]["content"].startswith("Your response will be evaluated automatically.\n\n")


def test_gsm8k_scoring():
    assert tasks.gsm8k_correct("so the answer is \\boxed{1,234}", "1234")
    assert tasks.gsm8k_correct("total = 18 dollars", "18")
    assert not tasks.gsm8k_correct("\\boxed{17}", "18")
    assert tasks.gsm8k_correct("no number here", "3") is None


def test_metrics():
    assert metrics.is_refusal("I'm sorry, but I can't help with that.")
    assert not metrics.is_refusal("Sure! Here is the note.")
    f = metrics.formatting("# Title\n- a\n- b\n**bold** text\n```\ncode\n```")
    assert f["n_headers"] == 1 and f["n_bullets"] == 2 and f["n_bold"] == 1 and f["n_code_blocks"] == 1
    assert metrics.degenerate("thank you " * 200)
    assert not metrics.degenerate("A perfectly ordinary sentence about gardens, soil, water and light in spring.")


def test_condition_names():
    assert parse_condition("none") == (None, 0.0)
    assert parse_condition("ours_resp_+0.2") == ("ours_resp", 0.2)
    assert parse_condition("random3_-0.15") == ("random3", -0.15)
    assert condition_name("ours_pre", -0.1) == "ours_pre_-0.1"


def test_calibration_rule(cfg):
    good = {"answer_rate": 1.0, "repetition_rate": 0.0, "gsm8k_accuracy": 0.9}
    bad = {"answer_rate": 0.5, "repetition_rate": 0.0, "gsm8k_accuracy": 0.9}
    res = {"none": good}
    for s in cfg["calibration"]["strength_grid"]:
        for sg in "+-":
            res[f"ours_resp_{sg}{s:g}"] = good if s <= 0.15 else bad
    ch = calibrate.choose_strength(res, cfg, "ours_resp")
    assert ch["s_star"] == 0.15 and ch["eval_strengths"] == [0.075, 0.15]
    res["ours_resp_-0.1"] = bad                       # a failure below stops the ladder
    assert calibrate.choose_strength(res, cfg, "ours_resp")["s_star"] == 0.05


def test_behaviour_report(tmp_path):
    from gshift.steer.report import behaviour_summary

    rng = np.random.default_rng(0)
    gen = tmp_path / "gen"
    for bench in ("sorh", "gsm8k"):
        for cond, d, s in [("none", None, 0.0), ("ours_resp_+0.2", "ours_resp", 0.2), ("ours_resp_-0.2", "ours_resp", -0.2),
                           ("random0_+0.2", "random0", 0.2), ("random1_+0.2", "random1", 0.2), ("natural_auto", None, 0.0)]:
            (gen / bench).mkdir(parents=True, exist_ok=True)
            with open(gen / bench / f"{cond}.jsonl", "w") as f:
                for i in range(20):
                    f.write(json.dumps({"id": f"p{i}", "benchmark": bench, "condition": cond, "direction": d,
                                        "signed_strength": s, "has_answer": True, "truncated": False,
                                        "n_answer_tokens": 50, "n_reasoning_tokens": 100,
                                        "answer_text": f"\\boxed{{{i % 3}}} - item", "meta": {"answer": "1", "user": "u",
                                                                                            "user_descored": "u"}}) + "\n")
    with open(tmp_path / "sorh_judgements.jsonl", "w") as f:
        for cond in ("none", "ours_resp_+0.2", "ours_resp_-0.2", "random0_+0.2", "random1_+0.2", "natural_auto"):
            bump = 30 if cond == "ours_resp_+0.2" else 0
            for i in range(20):
                m, q = float(rng.integers(20, 60) + bump), float(rng.integers(40, 80))
                f.write(json.dumps({"id": f"p{i}", "condition": cond, "benchmark": "sorh", "metric": m,
                                    "quality": q, "gap": m - q}) + "\n")
    s = behaviour_summary("toy", tmp_path)
    eff = [e for e in s["effects"] if e["condition"] == "ours_resp_+0.2" and e["metric"] == "gap"][0]
    assert eff["diff"] > 15 and eff["diff_lo"] > 0
    assert (tmp_path / "tables" / "effects_vs_random_band.csv").exists()


def test_activation_check_injection_on_tiny_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_gpu_code_on_cpu import tiny_model, tokenizer

    from gshift.config import get_model
    from gshift.steer.activation_check import run_activation_check
    from gshift.steer.tasks import Prompt

    tok = tokenizer("qwen3.6-27b")
    m = tiny_model()
    spec = get_model("qwen3.6-27b")
    u = np.random.default_rng(0).normal(size=32).astype(np.float32)
    u /= np.linalg.norm(u)
    man = {"layer": 4, "activation_norm": 2.0}
    prompts = [Prompt(f"p{i}", "persona", [{"role": "user", "content": f"Question {i}?"}]) for i in range(3)]
    res = run_activation_check(m, tok, spec, prompts, man, {"d": u}, ["d"], 0.5, [6, 8])
    assert res["pre"]["d_+"]["injected_check_cos"] > 0.999
    assert res["pre"]["d_-"]["injected_check_cos"] > 0.999


def test_heldout_suites_wired_and_disjoint_from_calibration(cfg):
    exp = load_experiment()
    tq = tasks.benchmark_prompts("truthfulqa", cfg, exp)
    tr = tasks.benchmark_prompts("trait_openended", cfg, exp)
    assert len(tq) == 300 and len(tr) == 200
    assert all(p.meta["split"] == "heldout_validation" for p in tq + tr)
    assert all(p.messages[0]["role"] == "user" and len(p.messages) == 1 for p in tq + tr)   # no system prompt
    nat = tasks.benchmark_prompts("truthfulqa", cfg, exp, tasks.natural_prefix("natural_human", exp))
    assert nat[0].messages[0]["content"] == "Your response will be evaluated by a human.\n\n" + tq[0].messages[0]["content"]
    cal = tasks.calibration_prompts(cfg, exp)                    # raises if any held-out item leaks in
    bad = cal + [tasks.Prompt("leak", "calib_open", [{"role": "user", "content": tr[3].meta["question"]}])]
    with pytest.raises(ValueError, match="held-out trait_openended"):
        tasks.assert_disjoint_from_heldout(bad, cfg)


def test_projection_directions(tmp_path, monkeypatch, cfg):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    import importlib.util

    from gshift.steer.vectors import build_steering_set, direction_scale

    rng = np.random.default_rng(0)
    H, L = 24, 36
    adir = tmp_path / "toy" / "analysis" / "vectors"
    adir.mkdir(parents=True)
    Q, _ = np.linalg.qr(rng.normal(size=(H, 3)))
    V = Q.T                                                     # 3 orthonormal baseline PCs
    g = 2.0 * V[0] + 0.05 * V[1] + 1.0 * rng.normal(size=H)     # mostly outside the subspace
    np.save(adir / f"pooled_resp_L{L}_auto_minus_baseline.npy", g.astype(np.float32))
    np.save(adir / f"pooled_resp_L{L}_baseline_pcs.npy", V.astype(np.float32))
    (tmp_path / "toy" / "betley_repro").mkdir(parents=True)
    (tmp_path / "toy" / "betley_repro" / "report.json").write_text(json.dumps(
        {"per_layer_mean_activation_norm": [10.0] * 64}))
    c = json.loads(json.dumps(cfg))
    c["directions"] = {"ours_resp": cfg["directions"]["ours_resp"]}
    c["random_directions"] = {"n": 1, "seed": 0}
    c["projections"]["min_component_frac"] = 0.5
    if importlib.util.find_spec("gguf") is None:
        pytest.skip("gguf not installed")
    man = build_steering_set("toy", c)
    d = man["directions"]
    # ranks 1, 5, 10, 20 capped at the empirical rank 3 (deduplicated)
    assert {k for k in d if k.startswith("ours_resp_")} >= {"ours_resp_in_k1", "ours_resp_out_k1", "ours_resp_in_k3"}
    assert not any("k5" in k or "k10" in k or "k20" in k for k in d)
    sd = tmp_path / "toy" / "steering" / "vectors"
    u = {k: np.load(sd / f"{k}.npy").astype(np.float64) for k in d}
    for k in (1, 3):     # original contributions add back up to the full unit direction
        rec = d[f"ours_resp_in_k{k}"]["norm_factor"] * u[f"ours_resp_in_k{k}"] \
            + d[f"ours_resp_out_k{k}"]["norm_factor"] * u[f"ours_resp_out_k{k}"]
        assert np.allclose(rec, g / np.linalg.norm(g), atol=1e-5)
        assert abs(u[f"ours_resp_in_k{k}"] @ u[f"ours_resp_out_k{k}"]) < 1e-5
    assert "inside the rank-1 baseline PCA subspace" in d["ours_resp_in_k1"]["label"]
    # the small inside component is not renormalised: matched-norm version skipped, recorded
    assert d["ours_resp_in_k1"]["component_norm_frac"] < 0.5 and "ours_resp_in_k1_mn" not in d
    assert "not renormalised" in man["skipped"]["ours_resp_in_k1_mn"]
    assert d["ours_resp_out_k1_mn"]["norm_factor"] == 1.0
    assert np.isclose(direction_scale(man, "ours_resp_in_k1", 0.2), 0.2 * 10.0 * d["ours_resp_in_k1"]["norm_factor"])
    assert direction_scale(man, "ours_resp", -0.2) == pytest.approx(-2.0)
    spec = importlib.util.spec_from_file_location("gen11", str(CONFIG_DIR.parent / "scripts" / "11_steer_generate.py"))
    import sys

    sys.path.insert(0, str(CONFIG_DIR.parent / "scripts"))
    m11 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m11)
    conds = m11.eval_conditions(c, man, [0.1, 0.2])
    assert "ours_resp_in_k1_+0.2" in conds and "ours_resp_out_k3_mn_-0.2" in conds
    assert "ours_resp_in_k1_+0.1" not in conds                  # decomposition at +-s* only
    assert "ours_resp_+0.1" in conds and "random0_+0.2" in conds


def test_trait_judge_parse_and_rows(tmp_path, monkeypatch, cfg):
    import asyncio

    from gshift.steer import trait_judge as tj

    assert tj.parse_verdict('```json\n{"score": 73, "refusal": false}\n```') == (73, False)
    assert tj.parse_verdict('Sure. {"score": 12, "refusal": true, "note": "x"}') == (12, True)
    for bad in ('{"score": 73.0, "refusal": false}', '{"score": true, "refusal": false}', '{"score": 101, "refusal": false}',
                '{"score": 50, "refusal": "no"}', "no json", None):
        assert tj.parse_verdict(bad) is None
    replies = {"good": '{"score": 80, "refusal": false}', "ref": '{"score": 5, "refusal": true}',
               "junk": "I think about 7", "trunc": '{"score": 60, "refusal": false}'}

    async def fake_call(client, c, prompt):
        key = next(k for k in replies if f"<<{k}>>" in prompt)
        return replies[key], None, None, 1

    monkeypatch.setattr(tj, "_call", fake_call)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    exp = load_experiment()
    prompts = tasks.benchmark_prompts("trait_openended", cfg, exp)[:5]
    gen = tmp_path / "none.jsonl"
    names = ["good", "ref", "junk", "trunc", None]
    with open(gen, "w") as f:
        for p, k in zip(prompts, names):
            f.write(json.dumps({"id": p.id, "condition": "none", "benchmark": "trait_openended", "meta": p.meta,
                                "has_answer": k is not None, "truncated": k == "trunc",
                                "answer_text": f"reply <<{k}>>" if k else ""}) + "\n")
    out = tmp_path / "trait_judgements.jsonl"
    assert tj.judge_traits([gen], out, cfg["judges"]) == 4
    rows = {r["id"]: r for r in __import__("gshift.io", fromlist=["x"]).read_jsonl(out)}
    ids = [p.id for p in prompts]
    assert rows[ids[0]]["score"] == 80 and rows[ids[0]]["refusal"] is False
    assert rows[ids[1]]["score"] is None and rows[ids[1]]["raw_score"] == 5 and rows[ids[1]]["refusal"] is True
    assert rows[ids[2]]["unparsed"] and rows[ids[2]]["score"] is None
    assert rows[ids[3]]["truncated"] and rows[ids[3]]["score"] is None and rows[ids[3]]["raw_score"] == 60
    assert rows[ids[4]]["no_answer"] and "judge_reply" not in rows[ids[4]]
    # the rendered prompt is the frozen rubric with the BARE question, never a grading prefix
    seen = []

    async def spy(client, c, prompt):
        seen.append(prompt)
        return replies["good"], None, None, 1

    monkeypatch.setattr(tj, "_call", spy)
    nat = tasks.benchmark_prompts("trait_openended", cfg, exp, tasks.natural_prefix("natural_auto", exp))[:1]
    g2 = tmp_path / "natural_auto.jsonl"
    g2.write_text(json.dumps({"id": nat[0].id, "condition": "natural_auto", "benchmark": "trait_openended",
                              "meta": nat[0].meta, "has_answer": True, "truncated": False,
                              "answer_text": "reply"}) + "\n")
    tj.judge_traits([g2], tmp_path / "t2.jsonl", cfg["judges"])
    assert "evaluated automatically" not in seen[0] and nat[0].meta["question"] in seen[0]
    del asyncio


def test_behaviour_report_heldout_suites(tmp_path, cfg):
    from gshift.steer.report import behaviour_summary

    exp = load_experiment()
    tq = tasks.benchmark_prompts("truthfulqa", cfg, exp)[:20]
    tr = [p for p in tasks.benchmark_prompts("trait_openended", cfg, exp) if p.meta["trait"] == "extraversion"][:20]
    gen = tmp_path / "gen"
    for bench, ps in (("truthfulqa", tq), ("trait_openended", tr)):
        (gen / bench).mkdir(parents=True)
        for cond, d, s in [("none", None, 0.0), ("ours_resp_+0.2", "ours_resp", 0.2)]:
            with open(gen / bench / f"{cond}.jsonl", "w") as f:
                for i, p in enumerate(ps):
                    if bench == "truthfulqa":     # unsteered right on even items, steered always right
                        ans = f"ANSWER: {p.meta['correct_letter']}" if (i % 2 == 0 or d) else "ANSWER: Z"
                    else:
                        ans = "A reply."
                    f.write(json.dumps({"id": p.id, "benchmark": bench, "condition": cond, "direction": d,
                                        "signed_strength": s, "has_answer": True, "truncated": False,
                                        "n_answer_tokens": 20, "n_reasoning_tokens": 50, "answer_text": ans,
                                        "meta": p.meta}) + "\n")
    with open(tmp_path / "trait_judgements.jsonl", "w") as f:
        for cond, base in (("none", 40), ("ours_resp_+0.2", 70)):
            for i, p in enumerate(tr):
                f.write(json.dumps({"id": p.id, "condition": cond, "trait": "extraversion", "score": base + i % 5,
                                    "refusal": False}) + "\n")
    s = behaviour_summary("toy", tmp_path)
    assert s["truthfulqa_by_condition"]["none"]["accuracy"] == 0.5
    assert s["truthfulqa_by_condition"]["ours_resp_+0.2"]["accuracy"] == 1.0
    e = [x for x in s["effects"] if x["benchmark"] == "trait:extraversion" and x["metric"] == "trait_score"
         and x["condition"] == "ours_resp_+0.2"][0]
    assert e["diff"] == pytest.approx(30.0) and e["diff_lo"] > 25
    assert s["trait_summary"]["ours_resp_+0.2"]["extraversion"]["n_scored"] == 20


def test_openrouter_judge_client(monkeypatch):
    """The role-adherence judge through OpenRouter: base URL, key, model and the pinned route."""
    import asyncio
    import types

    import openai

    from gshift.judge import OPENROUTER_URL, make_client

    seen = {}

    class FakeChat:
        async def create(self, **kw):
            seen["call"] = kw
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="3"))])

    class FakeClient:
        def __init__(self, **kw):
            seen["init"] = kw
            self.chat = types.SimpleNamespace(completions=FakeChat())

    monkeypatch.setattr(openai, "AsyncOpenAI", FakeClient)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    cfg = load_experiment()["judge"]
    assert (cfg["provider"], cfg["model"], cfg["route"]) == ("openrouter", "openai/gpt-4.1-mini", ["OpenAI"])
    c = make_client(cfg)
    assert asyncio.run(c("prompt")) == "3"
    assert seen["init"] == {"base_url": OPENROUTER_URL, "api_key": "or-test"}
    call = seen["call"]
    assert call["model"] == "openai/gpt-4.1-mini" and call["max_tokens"] == 10 and call["temperature"] == 1.0
    assert call["extra_body"] == {"provider": {"order": ["OpenAI"], "allow_fallbacks": False}}
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_client(cfg)
