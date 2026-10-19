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
