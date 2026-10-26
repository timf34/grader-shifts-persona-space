#!/usr/bin/env python
"""Does grading hide persona information or weaken it?  100-way persona-identity decoding on Q1 data.

Data: grader-shifts-persona-space/outputs/<model>/{baseline,auto,human}/activations/<role>/{pre,resp,reason}.npy
(float16 [n, 7 layers, 5120]) + index.jsonl (row-aligned metadata). Open-ended ('original') questions only;
the 100 prompted roles are the classes ('default' is excluded). Auto/human main sets = wording_set 'discovery'
(w1-w3, one record per (role, question, prompt variant), paired with baseline via baseline_id); the
held-out wordings v1/v2 (validation questions, prompt variant 0) are an extra test set under scheme A.

Decoders
  full : ridge classifier on standardised features (train mean/sd). Lambda chosen by exact grouped inner CV
         on the train set (hat-matrix block formula, one eigendecomposition per train set); probabilities =
         softmax(T * scores) with T fitted on the out-of-fold train scores.
  pc5 / pc20 : project raw activations onto the top-k PCs of the 100 baseline-TRAIN role centroids, then
         multinomial logistic regression (L2, alpha by grouped inner CV on train, criterion = CV log loss).
Hold-out schemes
  A : train discovery questions -> test validation questions (inner CV folds = question groups)
  B : train prompt variants {0,1,2} -> test {3,4} (inner CV folds = leave one prompt variant out)
Tests (per train decoder): base->base (i), base->auto (ii), auto->auto (iii), base->auto after subtracting the
mean auto-minus-baseline shift estimated on TRAIN data (ii_c); same for human. Length-matched variants restrict
to (role, question, variant) keys whose graded and baseline token counts (answer tokens for resp, reasoning
tokens for reason) are within a factor 1.3. CIs: percentile bootstrap over test questions (paired across tests).

Memory: one (model, layer, rep) slice at a time: 3 conditions x 100 roles x 150-170 rows x 5120 x 4 B = ~1 GB
float32, plus the ridge eigen-system in float64 (Z, Z@V: 2 x 10000 x 5120 x 8 B = 0.8 GB; Gram/eigvecs
2 x 0.2 GB) -> peak ~3 GB.  Run under: MallocLargeCache=0 ~/.claude/bin/memguard -m 8 -- python decoding.py ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import logsumexp

ROOT = Path("/Users/timf34/Documents/VSCode/grader-shifts-persona-space")
OUT = Path(__file__).resolve().parent
LAYERS = [16, 24, 32, 36, 40, 48, 56]
REPS = ["pre", "resp", "reason"]
CONDS = ["baseline", "auto", "human"]
LAM_REL = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]   # ridge lambda / n_train (mean Gram eigenvalue = n_train)
ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2]                # logistic L2 weight on mean NLL (1e-1 never near-best in a pilot slice)
LEN_RATIO = 1.3
B_BOOT = 2000


def roles_list() -> list[str]:
    import yaml
    cfg = yaml.safe_load(open(ROOT / "configs/experiment.yaml"))
    return [r for cat in cfg["roles"].values() for r in cat]


def load_meta(model: str):
    """Per condition: dict of numpy arrays over kept records (original questions, 100 prompted roles)."""
    roles = roles_list()
    assert len(roles) == 100 and len(set(roles)) == 100
    meta = {}
    for cond in CONDS:
        cols = {k: [] for k in ["role", "row", "q", "split", "pv", "wset", "has_answer", "n_ans", "n_reason", "key", "id"]}
        for ri, role in enumerate(roles):
            d = ROOT / "outputs" / model / cond / "activations" / role
            for r in map(json.loads, open(d / "index.jsonl")):
                if r["question_group"] != "original":
                    continue
                cols["role"].append(ri); cols["row"].append(r["row"]); cols["q"].append(r["question_id"])
                cols["split"].append(r["question_split"]); cols["pv"].append(r["prompt_variant"])
                cols["wset"].append(r["wording_set"]); cols["has_answer"].append(bool(r["has_answer"]))
                cols["n_ans"].append(r["n_answer_tokens"] or 0); cols["n_reason"].append(r["n_reasoning_tokens"] or 0)
                cols["key"].append(r["baseline_id"] if cond != "baseline" else r["id"]); cols["id"].append(r["id"])
        meta[cond] = {k: np.array(v) for k, v in cols.items()}
    return roles, meta


def load_acts(model: str, cond: str, roles, m, layer: int, rep: str) -> np.ndarray:
    li = LAYERS.index(layer)
    X = np.empty((len(m["role"]), 5120), np.float32)
    for ri, role in enumerate(roles):
        sel = np.flatnonzero(m["role"] == ri)
        arr = np.load(ROOT / "outputs" / model / cond / "activations" / role / f"{rep}.npy", mmap_mode="r")
        X[sel] = np.asarray(arr[m["row"][sel], li, :], dtype=np.float32)
        del arr
    return X


def valid_mask(m, rep):
    if rep == "pre":
        return np.ones(len(m["role"]), bool)
    if rep == "resp":
        return m["has_answer"] & (m["n_ans"] > 0)
    return m["n_reason"] > 0


# ----------------------------------------------------------------------------------------------- decoders
def nll_T(scores, y, T):
    s = T * scores
    return float(np.mean(logsumexp(s, axis=1) - s[np.arange(len(y)), y]))


def fit_ridge(X, y, groups, C=100):
    mu = X.mean(0, dtype=np.float64); sd = X.std(0, dtype=np.float64); sd[sd < 1e-6] = 1.0
    Z = (X - mu) / sd                      # float64
    n = len(y)
    Y = np.zeros((n, C)); Y[np.arange(n), y] = 1.0
    ybar = Y.mean(0); Yc = Y - ybar
    e, V = np.linalg.eigh(Z.T @ Z); e = np.clip(e, 0, None)
    P = Z @ V; del Z
    Q = P.T @ Yc
    gs = np.unique(groups)
    cv_res = {}
    best = None
    for lr in LAM_REL:
        D = 1.0 / (e + lr * n)
        PD = P * D
        Yhat = PD @ Q
        cv = np.empty_like(Yhat)
        for g in gs:
            idx = np.flatnonzero(groups == g)
            H = PD[idx] @ P[idx].T
            cv[idx] = Yc[idx] - np.linalg.solve(np.eye(len(idx)) - H, Yc[idx] - Yhat[idx])
        del PD, Yhat
        acc = float((cv.argmax(1) == y).mean())
        cv_res[lr] = acc
        if best is None or acc >= best[0]:    # ties -> larger lambda
            best = (acc, lr, cv + ybar)
    acc, lr, cvs = best
    r = minimize_scalar(lambda lt: nll_T(cvs, y, np.exp(lt)), bounds=(-3, 9), method="bounded")
    T = float(np.exp(r.x))
    D = 1.0 / (e + lr * n)
    W = (V @ (D[:, None] * Q)).astype(np.float32)
    return dict(kind="ridge", mu=mu.astype(np.float32), sd=sd.astype(np.float32), W=W, b=ybar.astype(np.float32),
                T=T, lam_rel=lr, cv_acc=cv_res, cv_nll=nll_T(cvs, y, T))


def predict(model, X):
    Z = (X - model["mu"]) / model["sd"]
    s = Z @ model["W"] + model["b"]
    if model["kind"] == "ridge":
        s = model["T"] * s
    return s - logsumexp(s, axis=1, keepdims=True)   # log-probs


def _lr_obj(w, Z, Y, alpha, C):
    k = Z.shape[1]
    W = w[: k * C].reshape(k, C); b = w[k * C:]
    s = Z @ W + b
    lse = logsumexp(s, axis=1, keepdims=True)
    Pm = np.exp(s - lse)
    n = len(Z)
    loss = float(np.sum(lse[:, 0] - np.sum(s * Y, 1)) / n + alpha * np.sum(W * W))
    G = (Pm - Y) / n
    gW = Z.T @ G + 2 * alpha * W
    return loss, np.concatenate([gW.ravel(), G.sum(0)])


def _lr_fit(Z, y, alpha, C, w0=None):
    Y = np.zeros((len(y), C)); Y[np.arange(len(y)), y] = 1
    w0 = np.zeros(Z.shape[1] * C + C) if w0 is None else w0
    r = minimize(_lr_obj, w0, args=(Z, Y, alpha, C), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=500, gtol=1e-5))
    return r.x


def fit_logreg(X, y, groups, C=100):
    mu = X.mean(0); sd = X.std(0); sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    k = Z.shape[1]
    gs = np.unique(groups)
    cv_res = {}
    wf = {g: None for g in gs}
    best = None
    for a in ALPHAS[::-1]:                     # strong -> weak for warm starts
        lp_all = np.empty((len(y), C))
        for g in gs:
            tr = groups != g
            wf[g] = _lr_fit(Z[tr], y[tr], a, C, wf[g])
            W = wf[g][: k * C].reshape(k, C); b = wf[g][k * C:]
            s = Z[~tr] @ W + b
            lp_all[~tr] = s - logsumexp(s, 1, keepdims=True)
        nll = float(-lp_all[np.arange(len(y)), y].mean()); acc = float((lp_all.argmax(1) == y).mean())
        cv_res[a] = dict(nll=nll, acc=acc)
        if best is None or nll < best[0]:
            best = (nll, a)
    a = best[1]
    w = _lr_fit(Z, y, a, C)
    return dict(kind="logreg", mu=mu, sd=sd, W=w[: k * C].reshape(k, C), b=w[k * C:], alpha=a,
                cv={str(kk): v for kk, v in cv_res.items()})


# ----------------------------------------------------------------------------------------------- bootstrap
def boot_tests(tests: dict, qnames, rng, deltas):
    """tests: name -> (qidx, correct, logp_correct). Percentile CIs over test questions (paired resamples)."""
    nq = len(qnames)
    B = rng.integers(0, nq, size=(B_BOOT, nq))
    W = np.zeros((B_BOOT, nq)); np.add.at(W, (np.arange(B_BOOT)[:, None], B), 1)   # resample multiplicities
    out, draws = {}, {}
    for name, (qi, cor, lp) in tests.items():
        n_q = np.bincount(qi, minlength=nq).astype(float)
        c_q = np.bincount(qi, weights=cor.astype(float), minlength=nq)
        p_q = np.bincount(qi, weights=np.exp(lp), minlength=nq)
        acc_b = (W @ c_q) / np.maximum(W @ n_q, 1)
        p_b = (W @ p_q) / np.maximum(W @ n_q, 1)
        draws[name] = (acc_b, p_b)
        out[name] = dict(n=int(len(cor)), acc=float(cor.mean()) if len(cor) else None,
                         acc_ci=[float(np.percentile(acc_b, 2.5)), float(np.percentile(acc_b, 97.5))],
                         p_correct=float(np.exp(lp).mean()) if len(cor) else None,
                         p_correct_ci=[float(np.percentile(p_b, 2.5)), float(np.percentile(p_b, 97.5))])
    dout = {}
    for a, b in deltas:
        if a in out and b in out:
            d_acc = draws[a][0] - draws[b][0]; d_p = draws[a][1] - draws[b][1]
            dout[f"{a} - {b}"] = dict(acc=out[a]["acc"] - out[b]["acc"],
                                      acc_ci=[float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
                                      p_correct=out[a]["p_correct"] - out[b]["p_correct"],
                                      p_correct_ci=[float(np.percentile(d_p, 2.5)), float(np.percentile(d_p, 97.5))])
    return out, dout


def spread_stats(Xp, y, C=100):
    """Between-role variance of centroids, mean within-role variance (summed over dims), and their ratio."""
    cents = np.stack([Xp[y == c].mean(0) for c in range(C)])
    between = float(((cents - cents.mean(0)) ** 2).sum(1).mean())
    within = float(((Xp - cents[y]) ** 2).sum(1).mean())
    return dict(between=between, within=within, ratio=between / within)


# ----------------------------------------------------------------------------------------------- one slice
def run_slice(model, layer, rep, roles, meta, rng, schemes=("A", "B"), spaces=("full", "pc5", "pc20")):
    t0 = time.time()
    X = {c: load_acts(model, c, roles, meta[c], layer, rep) for c in CONDS}
    print(f"[{model} L{layer} {rep}] loaded in {time.time()-t0:.0f}s", flush=True)
    v = {c: valid_mask(meta[c], rep) for c in CONDS}
    for c in CONDS:   # guard against empty / non-finite rows
        bad = ~np.isfinite(X[c]).all(1) | (np.abs(X[c]).sum(1) == 0)
        v[c] &= ~bad
    m = meta
    allq = sorted(set(m["baseline"]["q"]))
    qix = {q: i for i, q in enumerate(allq)}
    qi = {c: np.array([qix[q] for q in m[c]["q"]]) for c in CONDS}
    base_key_ix = {k: i for i, k in enumerate(m["baseline"]["key"])}
    ntok = "n_ans" if rep == "resp" else "n_reason"
    res = {}
    for scheme in schemes:
        def tr_te(c):
            mm = m[c]
            main = (mm["wset"] != "validation") if c != "baseline" else np.ones(len(mm["role"]), bool)
            if scheme == "A":
                tr = main & (mm["split"] == "discovery"); te = main & (mm["split"] == "validation")
            else:
                tr = main & np.isin(mm["pv"], [0, 1, 2]); te = main & np.isin(mm["pv"], [3, 4])
            return tr & v[c], te & v[c]
        sets = {c: tr_te(c) for c in CONDS}
        # inner-CV folds mirror the outer split: 4 folds of discovery questions (A) / leave-one-variant-out (B)
        disc = sorted(set(np.array(allq)[np.unique(qi["baseline"][m["baseline"]["split"] == "discovery"])]))
        qfold = np.array([disc.index(q) % 4 if q in disc else -1 for q in allq])
        grp = {c: (qfold[qi[c]] if scheme == "A" else m[c]["pv"]) for c in CONDS}
        # length-matched keys (graded record vs its paired baseline record)
        lenmatch = {}
        for c in ["auto", "human"]:
            te = sets[c][1]
            bi = np.array([base_key_ix.get(k, -1) for k in m[c]["key"]])
            ok = te & (bi >= 0)
            okb = np.zeros_like(ok)
            okb[ok] = sets["baseline"][1][bi[ok]]
            ok &= okb
            if rep != "pre":
                a_ = m[c][ntok].astype(float); b_ = np.where(bi >= 0, m["baseline"][ntok][np.maximum(bi, 0)], 0).astype(float)
                ratio = np.maximum(a_, b_) / np.maximum(np.minimum(a_, b_), 1)
                ok &= ratio <= LEN_RATIO
            bsel = np.zeros(len(m["baseline"]["role"]), bool); bsel[bi[ok]] = True
            lenmatch[c] = (ok, bsel)
        # shared translation (train data): graded mean - baseline mean
        shift = {c: X[c][sets[c][0]].mean(0) - X["baseline"][sets["baseline"][0]].mean(0) for c in ["auto", "human"]}
        # PCs of baseline-train role centroids (raw space)
        trb = sets["baseline"][0]; yb = m["baseline"]["role"]
        cents = np.stack([X["baseline"][trb & (yb == r)].mean(0, dtype=np.float64) for r in range(100)])
        gmean = cents.mean(0)
        _, sv, Vt = np.linalg.svd(cents - gmean, full_matrices=False)
        evr = (sv ** 2) / (sv ** 2).sum()
        res[scheme] = {"centroid_pc_var_explained": {"k5": float(evr[:5].sum()), "k20": float(evr[:20].sum())}}
        for space in spaces:
            t1 = time.time()
            if space == "full":
                proj = lambda A: A
                fit = fit_ridge
            else:
                k = int(space[2:]); Pk = Vt[:k].T.astype(np.float32)
                proj = lambda A, Pk=Pk: ((A - gmean.astype(np.float32)) @ Pk).astype(np.float64)
                fit = fit_logreg
            decs, tests = {}, {}
            for c in CONDS:
                tr = sets[c][0]
                decs[c] = fit(proj(X[c][tr]), m[c]["role"][tr], grp[c][tr])

            def ev(dec, c, sel, centered=False):
                A = X[c][sel]
                if centered:
                    A = A - shift[c]
                lp = predict(decs[dec], proj(A))
                y = m[c]["role"][sel]
                return qi[c][sel], lp.argmax(1) == y, lp[np.arange(len(y)), y]

            tests["i_base_on_base"] = ev("baseline", "baseline", sets["baseline"][1])
            for c in ["auto", "human"]:
                tests[f"ii_base_on_{c}"] = ev("baseline", c, sets[c][1])
                tests[f"iii_{c}_on_{c}"] = ev(c, c, sets[c][1])
                tests[f"iic_base_on_{c}_centered"] = ev("baseline", c, sets[c][1], centered=True)
                tests[f"x_{c}_on_base"] = ev(c, "baseline", sets["baseline"][1])
                ok, bsel = lenmatch[c]
                tests[f"len_{c}:i_base_on_base"] = ev("baseline", "baseline", bsel)
                tests[f"len_{c}:ii_base_on_{c}"] = ev("baseline", c, ok)
                tests[f"len_{c}:iii_{c}_on_{c}"] = ev(c, c, ok)
                tests[f"len_{c}:iic_base_on_{c}_centered"] = ev("baseline", c, ok, centered=True)
                if scheme == "A":   # held-out wordings v1/v2 (validation questions, pv 0)
                    vw = (m[c]["wset"] == "validation") & v[c]
                    tests[f"vw_{c}:ii_base_on_{c}"] = ev("baseline", c, vw)
                    tests[f"vw_{c}:iii_{c}_on_{c}"] = ev(c, c, vw)
            deltas = []
            for c in ["auto", "human"]:
                deltas += [(f"ii_base_on_{c}", "i_base_on_base"), (f"iii_{c}_on_{c}", "i_base_on_base"),
                           (f"iic_base_on_{c}_centered", "i_base_on_base"),
                           (f"len_{c}:ii_base_on_{c}", f"len_{c}:i_base_on_base"),
                           (f"len_{c}:iii_{c}_on_{c}", f"len_{c}:i_base_on_base")]
            tq = np.unique(np.concatenate([t[0] for t in tests.values() if len(t[0])]))
            remap = -np.ones(len(allq), int); remap[tq] = np.arange(len(tq))
            tests_r = {kk: (remap[t[0]], t[1], t[2]) for kk, t in tests.items() if len(t[0])}
            out, dout = boot_tests(tests_r, [allq[i] for i in tq], rng, deltas)
            info = {}
            for c, d in decs.items():
                info[c] = {kk: d[kk] for kk in ["lam_rel", "cv_acc", "T", "cv_nll", "alpha", "cv"] if kk in d}
                if "cv_acc" in info[c]:
                    info[c]["cv_acc"] = {str(a): b for a, b in info[c]["cv_acc"].items()}
            entry = dict(decoders=info, tests=out, deltas=dout)
            if space != "full":   # separability in the subspace (test data, raw projection)
                entry["spread_test"] = {c: spread_stats(proj(X[c][sets[c][1]]), m[c]["role"][sets[c][1]]) for c in CONDS}
            res[scheme][space] = entry
            g = lambda kk: out[kk]["acc"]
            print(f"  {scheme} {space:5s} i={g('i_base_on_base'):.3f} ii={g('ii_base_on_auto'):.3f} "
                  f"iii={g('iii_auto_on_auto'):.3f} iic={g('iic_base_on_auto_centered'):.3f} | human ii={g('ii_base_on_human'):.3f} "
                  f"iii={g('iii_human_on_human'):.3f} iic={g('iic_base_on_human_centered'):.3f}  ({time.time()-t1:.0f}s)", flush=True)
    res["n_valid"] = {c: int(v[c].sum()) for c in CONDS}
    return res


def check_alignment(model, roles):
    """index.jsonl row order vs .npy rows: shapes, `row` field, and norms.npy (written in the same extraction
    loop). pre: stored norm = ||vector|| (must match to fp16 precision). resp/reason: stored norm = MEAN per-token
    norm, so ||mean vector|| / stored must be <= 1 (+fp16 slack); we report the range and the across-row
    correlation of the two."""
    out = {"files": 0, "pre_max_rel_err": 0.0, "pool_ratio_max": 0.0, "pool_ratio_min": 9.0, "pool_corr_min": 1.0}
    for cond in CONDS:
        for role in roles + ["default"]:
            d = ROOT / "outputs" / model / cond / "activations" / role
            idx = [json.loads(line) for line in open(d / "index.jsonl")]
            assert all(r["row"] == i for i, r in enumerate(idx)), (cond, role)
            norms = np.load(d / "norms.npy", mmap_mode="r")
            assert norms.shape == (len(idx), 3, 7), (cond, role, norms.shape)
            for k, rep in enumerate(REPS):
                a = np.load(d / f"{rep}.npy", mmap_mode="r")
                assert a.shape == (len(idx), 7, 5120), (cond, role, rep, a.shape)
                nn = np.linalg.norm(np.asarray(a[:, 3, :], np.float32), axis=1)
                ref = np.asarray(norms[:, k, 3]); okm = ref > 0
                if rep == "pre":
                    out["pre_max_rel_err"] = max(out["pre_max_rel_err"], float((np.abs(nn - ref)[okm] / ref[okm]).max()))
                elif okm.sum() > 2:
                    r = nn[okm] / ref[okm]
                    out["pool_ratio_max"] = max(out["pool_ratio_max"], float(r.max()))
                    out["pool_ratio_min"] = min(out["pool_ratio_min"], float(r.min()))
                    out["pool_corr_min"] = min(out["pool_corr_min"], float(np.corrcoef(nn[okm], ref[okm])[0, 1]))
                out["files"] += 1
                del a
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--merge", action="store_true", help="merge parts/*.json into results.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[36])
    ap.add_argument("--reps", nargs="+", default=REPS)
    ap.add_argument("--schemes", nargs="+", default=["A", "B"])
    ap.add_argument("--spaces", nargs="+", default=["full", "pc5", "pc20"])
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.merge:
        res = {"settings": dict(lam_rel=LAM_REL, alphas=ALPHAS, len_ratio=LEN_RATIO, n_boot=B_BOOT, layers_index=LAYERS)}
        for f in sorted((OUT / "parts").glob("*.json")):
            mdl, rest = f.stem.split("_", 1)
            res.setdefault(mdl, {})[rest] = json.loads(f.read_text())
        (OUT / "results.json").write_text(json.dumps(res, indent=1))
        return
    roles, meta = load_meta(a.model)
    parts = OUT / "parts"; parts.mkdir(exist_ok=True)
    if a.check:
        r = check_alignment(a.model, roles)
        print(a.model, r)
        (parts / f"{a.model}_alignment.json").write_text(json.dumps(r))
        return
    for layer in a.layers:
        for rep in a.reps:
            rng = np.random.default_rng(20261026)
            r = run_slice(a.model, layer, rep, roles, meta, rng, a.schemes, a.spaces)
            (parts / f"{a.model}_L{layer}_{rep}.json").write_text(json.dumps(r))


if __name__ == "__main__":
    main()
