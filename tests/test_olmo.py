"""OLMo RL-stage comparison: design (configs/experiment_olmo_rl.yaml), model specs, and the
multi-token </think> split. Tokenizer tests skip when the OLMo tokenizer is not cached."""

import collections

import pytest

from gshift.config import REPO_ROOT, get_model, heldout_roles, load_experiment, statement
from gshift.design import build_requests

CFG = REPO_ROOT / "configs" / "experiment_olmo_rl.yaml"
KEYS = ("olmo3-32b-dpo", "olmo31-32b-rl100", "olmo31-32b-final")


def test_olmo_grid():
    exp = load_experiment(CFG)
    reqs = build_requests(exp, "main")
    assert len(reqs) == 50 * (30 * 4 + 30 * 6) + 60 * 5       # 50 roles x pv 0 + default control
    c = collections.Counter((r.question_group, r.condition) for r in reqs if r.role != "default")
    assert set(c.values()) == {50 * 30}
    assert {r.prompt_variant for r in reqs} == {0}
    assert len(heldout_roles(exp)) == 10
    assert {r.role for r in reqs if r.role_split == "heldout"} == set(heldout_roles(exp))
    # neutral sentences are prefixed exactly like the grading statements
    r = next(x for x in reqs if x.condition == "neutral_long")
    assert r.user_content.startswith(statement(exp, "neutral_long", r.wording_id) + "\n\n")


def test_env_override(monkeypatch):
    monkeypatch.setenv("GSHIFT_EXPERIMENT", "configs/experiment_olmo_rl.yaml")
    assert load_experiment()["design_name"] == "olmo_rl_v1"
    monkeypatch.delenv("GSHIFT_EXPERIMENT")
    assert "design_name" not in load_experiment()


def test_olmo_specs_share_tokenizer():
    specs = [get_model(k) for k in KEYS]
    assert all(s.revision and len(s.revision) == 40 for s in specs)
    assert len({(s.tokenizer_id, s.tokenizer_revision) for s in specs}) == 1
    assert len({s.revision for s in specs}) == 3
    assert all(s.n_layers == 64 and s.hidden_size == 5120 for s in specs)


@pytest.fixture(scope="module")
def olmo_tok():
    try:
        from transformers import AutoTokenizer

        s = get_model("olmo31-32b-final")
        return AutoTokenizer.from_pretrained(s.tokenizer_id, revision=s.tokenizer_revision, local_files_only=True)
    except Exception:
        pytest.skip("OLMo tokenizer not cached")


def test_multi_token_think_split(olmo_tok):
    from gshift.templates import answer_text, inspect_template, reasoning_text, split_generation

    info = inspect_template(olmo_tok, True, {})
    assert info.think_close_id is None and info.prompt_opens_think
    cases = {"Okay, let me think.\n</think>\n\nThe answer is 4.<|im_end|>": ("Okay, let me think.", "The answer is 4."),
             "reason x </think> y z": ("reason x", "y z")}
    for gen, (reas, ans) in cases.items():
        ids = olmo_tok.encode(gen, add_special_tokens=False)
        sp = split_generation(ids, olmo_tok, info)
        assert sp.think_closed and reasoning_text(ids, olmo_tok, sp) == reas and answer_text(ids, olmo_tok, sp) == ans
    ids = olmo_tok.encode("still thinking", add_special_tokens=False)
    sp = split_generation(ids, olmo_tok, info)
    assert not sp.think_closed and not sp.has_answer


def test_neutral_length_matched(olmo_tok):
    exp = load_experiment(CFG)
    n = lambda c: len(olmo_tok.encode(statement(exp, c, "w1"), add_special_tokens=False))  # noqa: E731
    assert abs(n("neutral") - n("auto")) <= 3
    assert abs(n("neutral_long") - n("auto_correct")) <= 3


def test_olmo_steering_plan(monkeypatch):
    """configs/steering_olmo.yaml: DPO-sourced directions, adherence over all 50 roles."""
    import importlib.util

    from gshift.config import load_yaml
    from gshift.steer.tasks import benchmark_prompts

    monkeypatch.setenv("GSHIFT_EXPERIMENT", "configs/experiment_olmo_rl.yaml")
    exp = load_experiment()
    cfg = load_yaml(REPO_ROOT / "configs" / "steering_olmo.yaml")
    assert {d.get("source_model") for d in cfg["directions"].values()} == {"olmo3-32b-dpo", "olmo31-32b-final"}
    adh = benchmark_prompts("adherence", cfg, exp)
    graded = benchmark_prompts("adherence_graded", cfg, exp)
    assert len(adh) == len(graded) == 50 * 10 and all(p.meta["group"] == "checkable" for p in adh)
    spec = importlib.util.spec_from_file_location("g", REPO_ROOT / "scripts" / "11_steer_generate.py")
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    man = {"directions": {**{d: {"kind": "recovered"} for d in cfg["directions"]},
                          "random0": {"kind": "random"}, "random1": {"kind": "random"}}}
    S = cfg["evaluation"]["strengths"]
    assert len(g.eval_conditions(cfg, man, S)) == 3 + 3 * 2 * 2 + 2 * 2 + 2 * 2   # + two final-sourced dirs at +-0.3
    assert all(c == "none" or "_-" in c for c in g.benchmark_conditions(cfg, man, S, "adherence_graded"))


def test_rl_episode_stage():
    """rl_episode: the persona instruction is embedded verbatim, the user message is the bare
    question, seeds match the main baseline, and the main grid / config id are untouched."""
    from gshift.design import role_instruction

    exp = load_experiment(CFG)
    reqs = build_requests(exp, "rl_episode")
    assert len(reqs) == 50 * 60 * 2 and {r.condition for r in reqs} == {"rl_env", "rl_deploy"}
    main = {r.id: r for r in build_requests(exp, "main")}
    for r in reqs:
        assert role_instruction(r.role, 0) in r.system_prompt and r.user_content == r.question
        assert r.baseline_id in main and main[r.baseline_id].seed == r.seed
    pair = {(r.role, r.question_id, r.condition): r.system_prompt for r in reqs}
    assert all(pair[(ro, q, "rl_env")] != pair[(ro, q, "rl_deploy")] for ro, q, c in pair if c == "rl_env")
    # the main grid is the one already generated: same requests as without the new block
    exp0 = {k: v for k, v in exp.items() if k != "system_conditions"}
    assert [r.id for r in build_requests(exp0, "main")] == list(main)
    spec = get_model("olmo31-32b-final")
    from gshift.generate import config_identity
    assert config_identity(spec, exp)[0] == config_identity(spec, exp0)[0]
