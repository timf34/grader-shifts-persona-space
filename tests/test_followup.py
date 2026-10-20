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
