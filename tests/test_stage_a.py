"""Q2 stage A: factorial directions, per-benchmark condition plans, sharding, adherence prompts
and summaries, steering-induced convergence. CPU only."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gshift.config import load_experiment, load_yaml  # noqa: E402
from gshift.steer.adherence import adherence_summary  # noqa: E402
from gshift.steer.tasks import benchmark_prompts  # noqa: E402
from gshift.steer.vectors import orthogonalise  # noqa: E402


def _gen_module():
    spec = importlib.util.spec_from_file_location("steer_generate", ROOT / "scripts" / "11_steer_generate.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _manifest():
    dirs = {d: {"kind": "recovered"} for d in
            ("ours_pre", "auto_human", "eval_only", "crit_only", "betley_matched", "betley_repro")}
    dirs.update({"random0": {"kind": "random"}, "random1": {"kind": "random"}})
    return {"directions": dirs}


def test_orthogonalise_removes_projection():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    r, frac = orthogonalise(a, [b])
    assert abs(r @ b) < 1e-9
    assert 0 < frac < 1
    assert np.isclose(frac, (a @ b) ** 2 / ((a @ a) * (b @ b)))


def test_stage_a_condition_plan_and_shards():
    g = _gen_module()
    cfg = load_yaml(ROOT / "configs" / "steering_stage_a.yaml")
    man = _manifest()
    S = cfg["evaluation"]["strengths"]
    allc = g.eval_conditions(cfg, man, S)
    assert len(allc) == 3 + 6 * 2 * 2 + 2 * 2 and len(set(allc)) == len(allc)
    adh = g.benchmark_conditions(cfg, man, S, "adherence")
    assert "natural_auto" not in adh and "none" in adh
    assert not any(c.startswith(("betley", )) for c in adh)
    graded = g.benchmark_conditions(cfg, man, S, "adherence_graded")
    assert all(c == "none" or "_-" in c for c in graded)          # subtract only
    desc = g.benchmark_conditions(cfg, man, S, "sorh_descored")
    assert not any(c.startswith("random") for c in desc)
    assert set(g.benchmark_conditions(cfg, man, S, "sorh")) == set(allc)
    # every condition goes to exactly one of m shards
    for m in (1, 3, 4):
        owners = [g.shard_of(c, allc, m) for c in allc]
        assert sorted(set(owners)) == list(range(m))


def test_adherence_prompts_paired_and_graded():
    exp = load_experiment()
    cfg = load_yaml(ROOT / "configs" / "steering_stage_a.yaml")
    plain = benchmark_prompts("adherence", cfg, exp)
    graded = benchmark_prompts("adherence_graded", cfg, exp)
    assert len(plain) == len(graded) == 400                     # 20 held-out roles x 10 validation q x 2 groups
    assert [p.id for p in plain] == [p.id for p in graded]       # shared ids = shared seeds
    for p, q in zip(plain, graded):
        assert p.messages[0] == q.messages[0]                    # same role system prompt
        assert q.messages[-1]["content"].endswith(p.messages[-1]["content"])
        assert q.messages[-1]["content"] != p.messages[-1]["content"]
        assert p.meta["question"] in p.messages[-1]["content"]
    chk = [q for q in graded if q.meta["group"] == "checkable"]
    assert chk and all(q.meta["graded_condition"] == "auto_correct" for q in chk)


def test_adherence_summary_paired_diff(tmp_path):
    rows = []
    for i in range(20):
        qid = "chk_x_01" if i % 2 else "orig_aa001"
        rows.append({"id": f"adh_r{i}", "condition": "adherence|none", "question_id": qid, "score": 3})
        rows.append({"id": f"adh_r{i}", "condition": "adherence|ours_pre_+0.3", "question_id": qid, "score": 1})
    p = tmp_path / "j.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    s = adherence_summary(p)
    e = s["adherence"]["ours_pre_+0.3"]["all"]
    assert e["n"] == 20 and np.isclose(e["diff_vs_none"], -2.0) and e["frac_fully"] == 0.0
    assert s["adherence"]["ours_pre_+0.3"]["checkable"]["n"] == 10


def test_steering_convergence_detects_planted_contraction(tmp_path, monkeypatch):
    from gshift.steer import activation_check as ac

    rng = np.random.default_rng(1)
    H, R, Q, k = 32, 12, 8, 4
    V = np.linalg.qr(rng.normal(size=(H, k)))[0].T                # k orthonormal persona axes
    vdir = tmp_path / "m" / "analysis" / "vectors"
    vdir.mkdir(parents=True)
    np.save(vdir / "original_pre_L48_baseline_pcs.npy", V)
    monkeypatch.setattr(ac, "model_dir", lambda key: tmp_path / key)
    pos = rng.normal(size=(R, k)) @ V * 5
    prompts, base, st = [], [], []
    for r in range(R):
        for q in range(Q):
            x = pos[r] + 0.05 * rng.normal(size=H)
            prompts.append(SimpleNamespace(meta={"role": f"r{r}", "question_id": f"orig_{q}", "group": "original"}))
            base.append([x, x])
            st.append([x + 1.0, (x + 1.0) * 0.6])                 # layer 36: shift only; layer 48: contraction
    base, st = np.array(base), np.array(st)
    out = ac.steering_convergence("m", prompts, base, st, [36, 48], [48], k=k)
    e = out["original_L48"]
    assert np.isclose(e["separation_ratio"], 0.6, atol=0.02)
    assert np.isclose(e["beta_crossfit"], -0.4, atol=0.05)


@pytest.mark.parametrize("model", ["qwen3.8-27b", "qwen3.6-27b"])
def test_stage_a_steering_set_builds(model):
    vj = ROOT / "outputs" / model / "steering" / "vectors.json"
    if not vj.exists():
        pytest.skip("steering set not built locally")
    man = json.loads(vj.read_text())
    assert {"ours_pre", "eval_only", "crit_only", "betley_matched"} <= set(man["directions"])
