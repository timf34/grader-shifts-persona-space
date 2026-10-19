"""The Q1 analysis recovers planted structure, keeps groups separate, distinguishes the three
geometric hypotheses, does not manufacture convergence from noise, and stays quiet under the null."""

from __future__ import annotations

import json

import numpy as np
import pytest

from gshift.analysis import stats as S
from gshift.analysis.cells import Cells
from gshift.analysis.contrast import disp, from_cells, permutation_null

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
    assert np.isclose(o["mean_loo_cos"], np.mean([S.cos(D[r], np.delete(D, r, 0).mean(0)) for r in range(6)]))
    n = np.linalg.norm(D, axis=1)
    C = (D @ D.T) / np.outer(n, n)
    assert np.isclose(o["mean_pairwise_cos"], C[~np.eye(6, dtype=bool)].mean())
    assert np.isclose(o["cos_g_v"], S.cos(g, v))
    s = np.linalg.svd(D, compute_uv=False)
    assert np.isclose(o["sv1_energy_frac"], s[0] ** 2 / (s**2).sum())


def test_contrast_matching_and_sign_flip():
    rng = np.random.default_rng(2)
    R, C, U, H = 2, 3, 4, 3
    N = np.ones((R, C, U))
    N[0, 2, 0] = 0                    # role 0 lacks condition c for unit 0
    Sx = (rng.normal(size=(R, C, U, H)) * N[..., None]).astype(np.float32)
    cells = Cells(["r0", "r1"], ["a", "b", "c"], [(f"q{i}", 0, 0) for i in range(U)], Sx, N)
    cd = from_cells(cells, "x", {"a": 0.5, "c": 0.5, "b": -1}, "g")
    assert cd.n_pairs().tolist() == [3, 4]
    want = (0.5 * Sx[0, 0, 1:] + 0.5 * Sx[0, 2, 1:] - Sx[0, 1, 1:]).mean(0)
    assert np.allclose(disp(cd)[0], want, atol=1e-5)
    # sign-flip null keeps |per-unit contrast| and only changes signs
    cd2 = from_cells(cells, "y", {"a": 1, "b": -1}, "g")
    D2 = disp(cd2)
    for n in permutation_null(cd2, 8, stat_fn=lambda Dp: {"d": Dp}):
        assert n["d"].shape == D2.shape


def test_planted_structure_recovered(tmp_path, monkeypatch, exp):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.report import analyze_model

    info = write_synthetic_model("toy", exp, stages=("main", "wording_validation", "followup"))
    (tmp_path / "toy" / "betley_repro").mkdir(parents=True)
    d = np.zeros((64, info["u_e"].size), np.float32)
    d[36] = info["u_e"]
    np.save(tmp_path / "toy" / "betley_repro" / "deltas_all_layers.npy", d)
    s = analyze_model("toy", exp, make_plots=True, reps=("pre", "resp"))
    res = s["results"]["resp"]["36"]
    o = res["original"]["auto_minus_baseline"]
    assert o["n_roles"] == 10 and o["observed"]["mean_loo_cos"] > 0.7
    assert o["null"]["g_norm"]["p_greater"] < 0.05
    for view in ("heldout_roles_x_discovery_questions", "discovery_roles_x_validation_questions",
                 "heldout_roles_x_validation_questions"):
        assert o["generalisation"][view]["cos_g"] > 0.9, view
    assert o["split_half"]["mean_cos"] > 0.9
    assert min(o["wording_stability"]["pairwise_cos"].values()) > 0.9
    assert res["original"]["auto_minus_human"]["cos_g_betley_repro"] > 0.95
    chk = res["checkable"]
    vec = tmp_path / "toy" / "analysis" / "vectors"
    ge = np.load(vec / "checkable_resp_L36_evaluator_main.npy")
    gc = np.load(vec / "checkable_resp_L36_criterion_main.npy")
    assert S.cos(ge, info["u_e"]) > 0.95 and S.cos(gc, info["u_c"]) > 0.95
    assert chk["evaluator_x_criterion"]["null"]["g_norm"]["p_greater"] > 0.01     # no planted interaction
    assert chk["evaluator_x_criterion"]["geometry"].get("skipped")               # not a two-state contrast
    assert res["original_vs_checkable_cos"]["automated_minus_human"] > 0.9
    assert "pooled" in res and "label" in res["pooled"]["auto_minus_baseline"]
    geo = o["geometry"]                                                          # translation world
    assert geo["verdict"]["H2_convergence"] != "supported"
    assert abs(geo["observed"]["separation_ratio"] - 1) < 0.15
    vw = res["validation_wordings"]
    assert vw["original|auto_minus_baseline|v1"]["cos_g_wording_g_discovery"] > 0.9
    assert res["followup"]["original"]["cos_g_original|auto_minus_baseline"] > 0.5
    assert "pc1_label" in res["original"]["persona_space"]["assistant_contrast"]
    out = tmp_path / "toy" / "analysis"
    assert (out / "figures" / "sweep_original_resp.png").exists()
    assert (vec / "pooled_resp_L36_auto_minus_baseline.npy").exists()
    assert (vec / "original_resp_L36_baseline_pcs.npy").exists()
    assert s["descriptives"]["missing_main"]["missing"] == 0
    json.loads((out / "summary.json").read_text())


def test_convergence_detected(tmp_path, monkeypatch, exp):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.report import analyze_model

    write_synthetic_model("conv", exp, alpha=0.3)
    s = analyze_model("conv", exp, make_plots=False, reps=("resp",), layers=[36], groups=["original"])
    geo = s["results"]["resp"]["36"]["original"]["auto_minus_baseline"]["geometry"]
    assert geo["observed"]["beta"] < -0.2
    assert geo["observed"]["separation_ratio"] < 0.7
    assert geo["verdict"]["H2_convergence"] == "supported"


def test_cross_fitting_prevents_spurious_convergence(tmp_path, monkeypatch, exp):
    """Pure translation, centroid noise comparable to persona spread: a naive same-data
    regression of displacement on baseline position would show strong 'convergence'."""
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.report import analyze_model

    write_synthetic_model("noisy", exp, alpha=0.0, persona_scale=0.3, noise=3.0, role_specific=0.0)
    s = analyze_model("noisy", exp, make_plots=False, reps=("resp",), layers=[36], groups=["original"])
    geo = s["results"]["resp"]["36"]["original"]["auto_minus_baseline"]["geometry"]
    assert abs(geo["observed"]["beta"]) < 0.25
    assert geo["verdict"]["H2_convergence"] != "supported"


def test_null_dataset_not_significant(tmp_path, monkeypatch, exp):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.report import analyze_model

    write_synthetic_model("null", exp, s_g=0, s_e=0, s_c=0, role_specific=0.0, noise=1.0, seed=5)
    s = analyze_model("null", exp, make_plots=False, reps=("resp",), layers=[36], groups=["original"])
    blk = s["results"]["resp"]["36"]["original"]["auto_minus_baseline"]
    assert blk["null"]["g_norm"]["p_greater"] > 0.01
    assert abs(blk["observed"]["mean_loo_cos"]) < 0.3


def test_crossmodel_comparison(tmp_path, monkeypatch, exp):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.crossmodel import compare_models
    from gshift.analysis.report import analyze_model

    write_synthetic_model("mA", exp, s_g=0.5, s_e=0.3, world_seed=0)
    write_synthetic_model("mB", exp, s_g=2.0, s_e=1.5, world_seed=1, H=40)
    for m in ("mA", "mB"):
        analyze_model(m, exp, make_plots=False, reps=("resp",), layers=[36])
    res = compare_models("mA", "mB", exp, n_boot=20)
    blk = res["primary"]["resp"]["original|auto_minus_baseline"]
    assert blk["paired_bootstrap_b_minus_a"]["g_norm_over_act_norm"]["lo"] > 0
    assert "rsa_persona_distances_baseline" in blk["second_order"]


def test_smoke_grid_analysis_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("GSHIFT_OUTPUTS", str(tmp_path))
    from gshift.analysis.report import analyze_model
    from gshift.config import load_experiment
    from gshift.smoke import smoke_experiment

    exp = smoke_experiment(load_experiment())
    write_synthetic_model("smoke", exp, stages=("main", "wording_validation", "followup"))
    s = analyze_model("smoke", exp, make_plots=True, reps=("pre", "resp"), layers=[36])
    assert s["results"]["resp"]["36"]["original"]["auto_minus_baseline"]["n_roles"] == 3
