"""The analysis recovers a planted shared shift, generalises where it should, and stays quiet
under the null."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gshift.analysis import stats as S
from gshift.analysis.cells import Cells, centroids

from synthetic import small_experiment, write_synthetic_model


@pytest.fixture
def exp():
    return small_experiment()


def test_gram_stats_match_direct_computation():
    rng = np.random.default_rng(0)
    D = rng.normal(size=(6, 20)) + 2 * rng.normal(size=20)
    v = rng.normal(size=20)
    o = S.gram_stats(D @ D.T, Dv={"v": D @ v}, vnorm={"v": np.linalg.norm(v)})
    g = D.mean(0)
    assert np.isclose(o["g_norm"], np.linalg.norm(g))
    loo = [S.cos(D[r], np.delete(D, r, 0).mean(0)) for r in range(6)]
    assert np.isclose(o["mean_loo_cos"], np.mean(loo))
    n = np.linalg.norm(D, axis=1)
    C = (D @ D.T) / np.outer(n, n)
    assert np.isclose(o["mean_pairwise_cos"], C[~np.eye(6, dtype=bool)].mean())
    assert np.isclose(o["cos_g_v"], S.cos(g, v))
    assert np.isclose(o["cos_g_unitavg"], S.cos(g, (D / n[:, None]).mean(0)))
    s = np.linalg.svd(D, compute_uv=False)
    assert np.isclose(o["sv1_energy_frac"], s[0] ** 2 / (s**2).sum())


def test_bootstrap_origin_excludes_self_pairs():
    rng = np.random.default_rng(1)
    D = rng.normal(size=(4, 10))
    Dd = D[[0, 0, 1, 2]]
    assert S.gram_stats(Dd @ Dd.T, origin=np.array([0, 0, 1, 2]))["mean_pairwise_cos"] < 0.99


def test_matched_view_drops_unpaired_units():
    rng = np.random.default_rng(2)
    R, C, U, H = 2, 2, 4, 3
    N = np.ones((R, C, U))
    N[0, 1, 0] = 0          # role 0 has no b-row for unit 0
    S_ = rng.normal(size=(R, C, U, H)).astype(np.float32) * N[..., None].astype(np.float32)
    cells = Cells(["r0", "r1"], ["a", "b"], [(f"q{i}", 0, 0) for i in range(U)], S_, N)
    mc = cells.matched("a", "b")
    assert mc.n_pairs().tolist() == [3, 4]
    mu = centroids(mc)
    assert np.allclose(mu[0, 0], S_[0, 0, 1:].mean(0), atol=1e-6)   # unit 0's a-row excluded too


def test_permutation_all_flips_negate():
    rng = np.random.default_rng(3)
    R, U, H = 3, 5, 7
    cells = Cells(["a", "b", "c"], ["x", "y"], [(f"q{i}", 0, 0) for i in range(U)],
                  rng.normal(size=(R, 2, U, H)).astype(np.float32), np.ones((R, 2, U)))
    D = S.disp(cells)
    Sa, Sb = cells.S[:, 0].astype(np.float64), cells.S[:, 1].astype(np.float64)
    assert np.allclose(((Sa.sum(1) + (Sb - Sa).sum(1)) - (Sb.sum(1) - (Sb - Sa).sum(1))) / U, -D, atol=1e-5)


def test_planted_shift_recovered(tmp_path, monkeypatch, exp):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.report import analyze_model

    info = write_synthetic_model("toy", exp, stages=("main", "paraphrase", "followup"))
    (tmp_path / "toy" / "betley_repro").mkdir(parents=True)
    d = np.zeros((64, info["shared"].size), np.float32)
    d[36] = info["shared"]
    np.save(tmp_path / "toy" / "betley_repro" / "deltas_all_layers.npy", d)
    s = analyze_model("toy", exp, make_plots=True)
    res = s["results"]["resp"]["36"]
    blk = res["pooled"]["auto_minus_baseline"]
    assert blk["n_roles"] == 10                         # discovery roles only
    assert blk["observed"]["mean_loo_cos"] > 0.7
    assert blk["null"]["g_norm"]["p_greater"] < 0.05
    assert blk["null"]["mean_loo_cos"]["p_greater"] < 0.05
    assert blk["cos_g_betley_repro"] > 0.95
    assert blk["split_half"]["mean_cos"] > 0.9
    gen = blk["generalisation"]
    for view in ("heldout_roles_x_discovery_questions", "discovery_roles_x_validation_questions",
                 "heldout_roles_x_validation_questions"):
        assert gen[view]["cos_g"] > 0.9, view
        assert gen[view]["null"]["p_greater"] < 0.05, view
    assert blk["default_assistant"]["cos_g_roles"] > 0.8
    assert "insufficient_roles" in blk["adherence_filtered"]
    assert res["original_vs_checkable_cos"]["auto_minus_baseline"] > 0.9
    assert res["paraphrases"]["auto_p1"]["cos_g_para_g_discovery"] > 0.9
    assert res["paraphrases"]["auto_p1"]["cos_heldout_roles_g_discovery"] > 0.9
    assert res["followup"]["eval_awareness"]["cos_g_auto_minus_baseline"] > 0.9
    assert res["pooled"]["persona_space"]["auto_minus_baseline"]["captured_frac"][39] <= 1.0
    out = tmp_path / "toy" / "analysis"
    assert (out / "figures" / "sweep_pooled_resp.png").exists()
    assert (out / "vectors" / "pooled_resp_L36_auto_minus_baseline.npy").exists()
    assert s["descriptives"]["missing_main"]["missing"] == 0
    json.loads((out / "summary.json").read_text())


def test_group_specific_component_detected(tmp_path, monkeypatch, exp):
    """A shift that differs between original and checkable questions shows up as reduced alignment."""
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.report import analyze_model

    write_synthetic_model("grp", exp, shift_scale=1.0, checkable_extra=2.0)
    s = analyze_model("grp", exp, make_plots=False, reps=("resp",), layers=[36])
    assert s["results"]["resp"]["36"]["original_vs_checkable_cos"]["auto_minus_baseline"] < 0.7


def test_null_dataset_not_significant(tmp_path, monkeypatch, exp):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.report import analyze_model

    write_synthetic_model("null", exp, shift_scale=0.0, role_specific=0.0, noise=1.0, seed=5)
    s = analyze_model("null", exp, make_plots=False, reps=("resp",), layers=[36], groups=["pooled"])
    blk = s["results"]["resp"]["36"]["pooled"]["auto_minus_baseline"]
    assert blk["null"]["g_norm"]["p_greater"] > 0.01
    assert abs(blk["observed"]["mean_loo_cos"]) < 0.3


def test_crossmodel_comparison(tmp_path, monkeypatch, exp):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.crossmodel import compare_models
    from gshift.analysis.report import analyze_model

    write_synthetic_model("mA", exp, shift_scale=0.5, world_seed=0)
    write_synthetic_model("mB", exp, shift_scale=2.0, world_seed=1, H=40)
    for m in ("mA", "mB"):
        analyze_model(m, exp, make_plots=False, reps=("resp",), layers=[36], groups=["pooled"])
    res = compare_models("mA", "mB", exp, n_boot=30)
    pb = res["primary"]["resp"]["paired_bootstrap_b_minus_a"]["auto_minus_baseline"]
    assert pb["g_norm_over_act_norm"]["lo"] > 0
    assert pb["mean_loo_cos"]["mean_diff"] > 0
    assert "rsa_persona_distances_baseline" in res["primary"]["resp"]["auto_minus_baseline"]["second_order"]
