"""The Gram-based cross-checkpoint statistics equal their direct definitions."""

import numpy as np

from gshift.analysis import rl_compare as rc
from gshift.analysis.geometry import _pairwise_cross
from gshift.analysis.stats import gram_stats


def _data(R=12, H=30, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(R, H)) + rng.normal(size=H)


def test_shift_stats_match_gram_stats_with_duplicates():
    D = _data()
    G = D @ D.T
    idx = rc.boot_idx(len(D), 5, 1)
    g, loo = rc.shift_stats(G, np.vstack([np.arange(len(D)), idx]))
    ref0 = gram_stats(G)
    assert np.isclose(g[0], ref0["g_norm"]) and np.isclose(loo[0], ref0["mean_loo_cos"])
    for b, ix in enumerate(idx, start=1):
        Db = D[ix]
        ref = gram_stats(Db @ Db.T, origin=ix)
        assert np.isclose(g[b], ref["g_norm"]) and np.isclose(loo[b], ref["mean_loo_cos"])


def test_mean_pair_is_mean_pairwise_squared_distance():
    X = _data()
    S = rc.pair_cross(X)
    m = rc.mean_pair(S, rc.ident(len(X)))[0]
    d = [np.sum((X[i] - X[j]) ** 2) for i in range(len(X)) for j in range(i + 1, len(X))]
    assert np.isclose(m, np.mean(d))
    Y = _data(seed=3)
    assert np.isclose(rc.mean_pair(rc.pair_cross(X, Y), rc.ident(len(X)))[0], _pairwise_cross(X - X.mean(0), Y - Y.mean(0)))
    ix = np.array([[0, 0, 1, 2, 3, 3, 4, 5, 6, 7, 8, 9]])        # duplicates: same-origin pairs excluded
    Xb = X[ix[0]]
    d = [np.sum((Xb[i] - Xb[j]) ** 2) for i in range(12) for j in range(i + 1, 12) if ix[0, i] != ix[0, j]]
    assert np.isclose(rc.mean_pair(S, ix)[0], np.mean(d))


def test_landscape_from_gram_matches_direct():
    X = _data(R=15, H=40)
    d = X.mean(0) + np.random.default_rng(5).normal(size=40) * 3
    for ix in (np.arange(15), rc.boot_idx(15, 1, 2)[0]):
        c = X.mean(0)
        Xc, dc = X - c, d - c
        o = rc.landscape_from_gram(Xc @ Xc.T, Xc @ dc, float(dc @ dc), ix[None])
        B = X[ix]
        Bc = B - B.mean(0)
        ev = np.clip(np.linalg.eigvalsh(Bc @ Bc.T)[::-1], 0, None)
        p = rc.pca(B)
        a = d - B.mean(0)
        assert np.isclose(o["var"][0], (Bc**2).sum(1).mean())
        assert np.isclose(o["pr"][0], ev.sum() ** 2 / (ev**2).sum())
        assert np.isclose(o["pc1_share"][0], ev[0] / ev.sum())
        assert np.isclose(o["abs_cos_pc1_assistant"][0], abs(p["components"][0] @ a) / np.linalg.norm(a))
        assert np.isclose(o["default_dist_over_spread"][0], np.linalg.norm(a) / np.sqrt((Bc**2).sum(1).mean()))


def test_diff_summary_and_twoway():
    a, b = np.zeros(100), np.full(100, 2.0)
    d = rc.diff_summary(0.0, 2.0, a, b)
    assert d["est"] == 2.0 and d["excludes_0"]
    V = np.ones((4, 5))
    V[0, 0] = np.nan
    assert np.allclose(rc.twoway_boot_mean(V, 20, 0), 1.0)
    assert np.allclose(rc.unit_boot_mean(np.ones(7), 10, 0), 1.0)


def test_jackknife_of_mean_is_standard_error():
    x = np.random.default_rng(1).normal(size=25)
    jv = np.array([np.delete(x, i).mean() for i in range(len(x))])
    j = rc.jack_summary(x.mean(), jv)
    assert np.isclose(j["se_jack"], x.std(ddof=1) / np.sqrt(len(x)))
    assert np.array_equal(rc.jack_idx(4)[1], [0, 2, 3])
