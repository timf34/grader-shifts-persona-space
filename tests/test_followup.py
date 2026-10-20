"""Follow-up A/B code on CPU: frozen panel integrity, executable shortcut scoring, the
cross-reading plan and its decomposition identities (through the real extractor, tiny model)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gshift import DATA_DIR
from gshift.config import load_experiment
from gshift.followup.crossread import decompose, plan_cells, select_roles
from gshift.followup.shortcut_score import score, static_special_case
from gshift.followup.shortcut_tasks import CONDITIONS, panel_sha, special_case_solution, user_message

PANEL = DATA_DIR / "followup" / "shortcut_panel_v1.jsonl"


@pytest.fixture(scope="module")
def panel():
    return [json.loads(l) for l in open(PANEL)]


def test_panel_frozen_and_audited(panel):
    prov = json.loads((DATA_DIR / "followup" / "PROVENANCE_shortcut_panel_v1.json").read_text())
    assert panel_sha(panel) == prov["panel_sha256"]
    assert len(panel) == 40 and sum(t["split"] == "discovery" for t in panel) == 20
    assert all(t["audit"]["ok"] for t in panel)
    for t in panel:
        assert t["visible_tests"][:-1] == t["original_visible_tests"][:-1]
        assert t["visible_tests"][-1] != t["original_visible_tests"][-1]
        assert t["corrupted_value"] != t["true_value"]
    assert user_message(panel[0], "baseline").startswith(panel[0]["prompt"].strip())
    assert user_message(panel[0], "blind_spot").startswith(CONDITIONS["blind_spot"] + "\n\n")


def test_executable_scoring(panel):
    t = panel[0]
    assert score(f"```python\n{t['reference_code']}\n```", "", t)["outcome"] == "CORRECT"
    hack = score(f"```python\n{special_case_solution(t)}\n```", "", t)
    assert hack["outcome"] == "SHORTCUT" and hack["static_special_case"] and not hack["hidden_pass"]
    assert score("no code here", "", t)["outcome"] == "NO_CODE"
    # a model that appends the (corrupted) asserts to its own code must not be scored as crashing
    ref_with_asserts = f"```python\n{t['reference_code']}\n" + "\n".join(t["visible_tests"]) + "\n```"
    assert score(ref_with_asserts, "", t)["outcome"] == "CORRECT"
    assert not static_special_case(t["reference_code"], t)


def test_decomposition_identities():
    rng = np.random.default_rng(0)
    v = {(a, b): rng.normal(size=8) for a in ("baseline", "graded") for b in ("baseline", "graded")}
    d = decompose(v)
    assert np.allclose(d["Delta"], d["CE_B"] + d["TE_c"]) and np.allclose(d["Delta"], d["TE_base"] + d["CE_G"])
    assert np.allclose(d["I"], d["CE_G"] - d["CE_B"])


def test_role_selection_is_stratified_and_never_heldout():
    exp = load_experiment()
    from gshift.config import heldout_roles

    complete = {r for rs in exp["roles"].values() for r in rs}
    sel, short = select_roles(exp, complete)
    assert len(sel) == 20 and not short and not (set(sel) & set(heldout_roles(exp)))
    sel2, short2 = select_roles(exp, set(sel[:6]))
    assert sel2 == sel[:6][: len(sel2)] or set(sel2) <= set(sel[:6])
    assert sum(short2.values()) == 20 - len(sel2)


def test_cross_reading_cells_through_extractor():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_gpu_code_on_cpu import tiny_model, tokenizer

    from gshift.design import build_requests
    from gshift.extract import Extractor
    from gshift.templates import prompt_token_ids

    exp = load_experiment()
    tok = tokenizer("qwen3.6-27b")
    m = tiny_model()
    render = lambda msgs: prompt_token_ids(tok, msgs, True)  # noqa: E731
    role = "mentor"
    qs = exp["pilot"]["questions"][:2] + exp["pilot"]["questions"][10:11]     # 2 original + 1 checkable
    exp = dict(exp, pilot=dict(exp["pilot"], questions=qs))
    recs = {}
    for r in build_requests(exp, "main", [role], include_default=False):
        if r.prompt_variant != 0 or r.question_id not in qs:
            continue
        body = tok(f"think {r.condition}\n</think>\n\nanswer {r.condition} {r.question_id}.",
                   add_special_tokens=False)["input_ids"] + [248046]
        recs.setdefault(r.condition, {})[r.id] = {"id": r.id, "prompt_ids": render(list(r.messages)), "gen_ids": body,
                                                  "has_answer": True}
    for c in ("baseline", "auto", "human", "auto_correct", "human_correct", "auto_quality", "human_quality"):
        recs.setdefault(c, {})
    jobs, probs = plan_cells(exp, role, recs, render)
    assert not probs and len(jobs) == 2 * 7 + 1 * 13
    ex = Extractor(m, tok, [2, 4], True)
    by = {}
    for j in jobs:
        by[(j["question_id"], j["condition"], j["text"], j["read"])] = ex.record(
            {"prompt_ids": j["prompt_ids"], "gen_ids": j["gen_ids"]})["acts"]["resp"]
    q = qs[0]
    bb = by[(q, "baseline", "baseline", "baseline")]
    # own-context graded cell equals extracting the original graded record directly
    gid = next(i for i in recs["auto"] if f"|{q}|" in i)
    direct = ex.record(recs["auto"][gid])["acts"]["resp"]
    assert np.allclose(by[(q, "auto", "graded", "graded")], direct, atol=1e-5)
    # reading the SAME text under another context changes the representation (context effect != 0)
    assert not np.allclose(by[(q, "auto", "baseline", "graded")], bb)
    d = decompose({("baseline", "baseline"): bb, ("baseline", "graded"): by[(q, "auto", "baseline", "graded")],
                   ("graded", "baseline"): by[(q, "auto", "graded", "baseline")],
                   ("graded", "graded"): by[(q, "auto", "graded", "graded")]})
    assert np.allclose(d["Delta"], d["CE_B"] + d["TE_c"])


def test_transplant_implementation_checks():
    """Same-context donor and lambda=0 are no-ops; at lambda=1 the edited host coordinate
    equals the donor's (donor-matched coordinate replacement)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_gpu_code_on_cpu import tiny_model

    from gshift.followup.transplant import TransplantConfig, generate_with_transplant, random_like
    from gshift.hooks import capture_residual

    m = tiny_model(vocab=512)
    torch.manual_seed(0)
    host = torch.randint(0, 512, (9,)).tolist()
    donor = host[:3] + torch.randint(0, 512, (6,)).tolist()          # different context, same length
    U = random_like(np.zeros((2, 32)), seed=1)
    plain, _ = generate_with_transplant(m, host, host, TransplantConfig(layer=4, U=U, mode="none"), 6)
    same, _ = generate_with_transplant(m, host, host, TransplantConfig(layer=4, U=U, lam=1.0), 6)
    assert same == plain
    zero, _ = generate_with_transplant(m, host, donor, TransplantConfig(layer=4, U=U, lam=0.0), 6)
    assert zero == plain
    cfg = TransplantConfig(layer=4, U=U, lam=1.0)
    gen, log = generate_with_transplant(m, host, donor, cfg, 1)
    # recompute the first step's donor coordinate and the edited host coordinate
    with torch.no_grad(), capture_residual(m, [4]) as st:
        m(input_ids=torch.tensor([donor]))
    c_d = torch.tensor(U, dtype=torch.float32) @ st[4][0, -1].float()
    assert np.allclose(log[0]["donor"], c_d.numpy(), atol=1e-4)
    big, _ = generate_with_transplant(m, host, donor, TransplantConfig(layer=4, U=U, lam=50.0), 6)
    rel, _ = generate_with_transplant(m, host, donor, TransplantConfig(layer=4, U=U, lam=50.0, release_after=2,
                                                                       recompute_cache_on_release=True), 6)
    assert len(big) == len(rel) == 6


def test_selfrating_axis_recovers_planted_direction():
    from gshift.followup.selfrating import Standardiser, context_by_strategy, cut_points, fit_axis, parse_rating, validate_axis

    rng = np.random.default_rng(0)
    H, n = 40, 600
    u_true = rng.normal(size=H)
    u_true /= np.linalg.norm(u_true)
    types = rng.integers(0, 3, n)
    r = rng.integers(0, 101, n).astype(float)
    X = rng.normal(size=(n, H)) + np.outer((r - 50) / 25, u_true) + 3 * np.eye(3)[types] @ rng.normal(size=(3, H))
    X[:, 0] *= 100                                                    # a sink coordinate
    disc, held = np.arange(n) < 400, np.arange(n) >= 400
    std = Standardiser.fit(X[disc])
    assert not std.keep[0]
    u = fit_axis(X[disc], r[disc], types[disc], std)
    v = validate_axis(X[held], r[held], std, u)
    assert v["validated"] and v["spearman"] > 0.3
    planted = (u_true / std.sd) * std.keep                     # the planted direction in standardised space
    assert abs(float(u @ planted / np.linalg.norm(planted))) > 0.6
    assert parse_rating(" 85.") == 85 and parse_rating("none") is None and parse_rating("250") is None
    assert len(cut_points(900)) == 8 and cut_points(5) == [1, 2, 3, 4]
    s = {("legit", "base"): np.zeros(30), ("shortcut", "base"): np.zeros(30) - 1,
         ("legit", "graded"): np.zeros(30), ("shortcut", "graded"): np.ones(30)}
    assert context_by_strategy(s)["interaction"] == 2.0
