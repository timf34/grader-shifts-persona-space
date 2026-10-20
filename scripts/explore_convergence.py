#!/usr/bin/env python3
"""Exploratory follow-ups on the grading shift, from the saved per-role condition centroids
(<analysis>/vectors/{group}_{rep}_L{L}_condition_centroids.npz; no activations loaded).

1. Convergence point. In the top-k baseline persona PCs, regress each discovery role's shift
   D_r on its baseline position x_r with one isotropic slope: D_r = a + beta * x_r. The fixed
   point c = -a / beta is where a persona would not move (apart from the shared component).
   Cross-fitted: x from one question split, D from the other (both directions, averaged), so
   measurement noise in x cannot fake contraction. Reports c's nearest baseline roles and the
   default Assistant, and how far apart the conditions' fixed points are. Role bootstrap.
2. Shift-space PCA. PCA of the role displacements (discovery questions) after removing the
   shared shift; reliability = correlation of role scores with the other split's displacements;
   extreme roles per PC; alignment with baseline persona PCs.
3. Grading space. SVD of the condition-mean shifts (all conditions vs baseline): how many
   dimensions do the grading conditions span, and what do the axes separate?

    python scripts/explore_convergence.py --model qwen3.8-27b [--rep pre] [--layer 36] [--k 10]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gshift.config import heldout_roles, load_experiment, role_categories  # noqa: E402
from gshift.io import outputs_root  # noqa: E402


def vec_dir(model: str) -> Path:
    merged = outputs_root() / model / "analysis" / "vectors"
    return merged if merged.exists() else outputs_root() / model / "analysis_parts"


def load_centroids(model: str, group: str, rep: str, L: int):
    name = f"{group}_{rep}_L{L}_condition_centroids.npz"
    d = vec_dir(model)
    path = d / name if (d / name).exists() else next(d.glob(f"{rep}_L{L}/vectors/{name}"))
    z = np.load(path)
    return [str(r) for r in z["roles"]], [str(c) for c in z["conds"]], {k: z[k].astype(np.float64) for k in z.files
                                                                         if k.startswith("mu")}


def pca(X: np.ndarray, k: int):
    mu = X.mean(0)
    U, s, Vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, Vt[:k], (s ** 2) / (s ** 2).sum()


def fixed_point(Xk: np.ndarray, Dk: np.ndarray):
    """OLS of D on x with one isotropic slope: D = a + beta*x. Returns (c, beta)."""
    xc, dc = Xk - Xk.mean(0), Dk - Dk.mean(0)
    beta = float((xc * dc).sum() / (xc * xc).sum())
    a = Dk.mean(0) - beta * Xk.mean(0)
    return -a / beta, beta


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--rep", default="pre")
    p.add_argument("--layer", type=int, default=36)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--n-boot", type=int, default=500)
    args = p.parse_args()
    exp = load_experiment()
    cats = role_categories(exp)
    held = set(heldout_roles(exp))
    rng = np.random.default_rng(0)
    out = {"model": args.model, "rep": args.rep, "layer": args.layer, "k": args.k}

    for group in ("original", "checkable"):
        roles, conds, mu = load_centroids(args.model, group, args.rep, args.layer)
        disc = np.array([r not in held for r in roles])
        ib = conds.index("baseline")
        graded = [c for c in conds if c != "baseline"]
        Xd, Xv = mu["mu_discovery"][:, ib], mu["mu_validation"][:, ib]
        # persona PCA on discovery-role baseline centroids (discovery questions), as in the main analysis
        center, V, evr = pca(Xd[disc], args.k)
        proj = lambda A: (A - center) @ V.T  # noqa: E731
        res = {"baseline_pc_var": evr[: args.k].round(4).tolist(), "conditions": {}}
        dflt = {s: mu[f"mu_default_{s}"] for s in ("discovery", "validation") if f"mu_default_{s}" in mu}
        allXk = proj(Xd)  # all 100 roles' baseline positions (discovery questions)
        scale = float(np.sqrt(((allXk[disc] - allXk[disc].mean(0)) ** 2).sum(1).mean()))  # rms persona spread
        fps = {}
        for c in graded:
            ic = conds.index(c)
            Dd = mu["mu_discovery"][:, ic] - Xd
            Dv = mu["mu_validation"][:, ic] - Xv
            idx = np.where(disc)[0]

            def cross(sel):
                # x from discovery questions, D from validation questions, and the reverse
                c1, b1 = fixed_point(proj(Xd[sel]), Dv[sel] @ V.T)
                c2, b2 = fixed_point(proj(Xv[sel]), Dd[sel] @ V.T)
                return (c1 + c2) / 2, (b1 + b2) / 2

            cfp, beta = cross(idx)
            boots = [cross(rng.choice(idx, size=len(idx), replace=True)) for _ in range(args.n_boot)]
            cb = np.array([b[0] for b in boots])
            bb = np.array([b[1] for b in boots])
            dist = np.sqrt(((allXk - cfp) ** 2).sum(1))
            order = np.argsort(dist)
            entry = {
                "beta_crossfit": round(beta, 3), "beta_ci95": np.percentile(bb, [2.5, 97.5]).round(3).tolist(),
                "fixed_point_pc": cfp[:3].round(2).tolist(),
                "fixed_point_spread_over_persona_rms": round(float(np.sqrt(((cb - cfp) ** 2).sum(1).mean())) / scale, 3),
                "nearest_roles": [(roles[i], cats.get(roles[i], "?"), round(float(dist[i]) / scale, 2)) for i in order[:6]],
                "fixed_point_to_centroid_over_rms": round(float(np.linalg.norm(cfp - allXk[disc].mean(0))) / scale, 2),
            }
            if dflt:
                dA = proj(dflt["discovery"][ib][None])[0]
                entry["default_assistant_to_fixed_point_over_rms"] = round(float(np.linalg.norm(dA - cfp)) / scale, 2)
                entry["default_assistant_rank_among_roles"] = int((dist < np.linalg.norm(dA - cfp)).sum())
            res["conditions"][c] = entry
            fps[c] = cb
        # are the conditions' fixed points the same place? (distance between bootstrap means, in persona rms)
        names = list(fps)
        res["fixed_point_distances_over_rms"] = {
            f"{a}|{b}": round(float(np.linalg.norm(fps[a].mean(0) - fps[b].mean(0))) / scale, 3)
            for i, a in enumerate(names) for b in names[i + 1:]}

        # 2. shift-space PCA for the most RLVR-like condition of the group
        c = "auto" if group == "original" else "auto_correct"
        ic = conds.index(c)
        Dd = (mu["mu_discovery"][:, ic] - Xd)[disc]
        Dv = (mu["mu_validation"][:, ic] - Xv)[disc]
        dmu, W, devr = pca(Dd, 5)
        sd, sv = (Dd - dmu) @ W.T, (Dv - Dv.mean(0)) @ W.T
        rr = [r for r, d in zip(roles, disc) if d]
        pcs = []
        for j in range(5):
            o = np.argsort(sd[:, j])
            pcs.append({
                "var_frac": round(float(devr[j]), 3),
                "split_reliability_r": round(float(np.corrcoef(sd[:, j], sv[:, j])[0, 1]), 3),
                "max_abs_cos_with_baseline_pc": round(float(np.abs(V @ W[j]).max()), 3),
                "best_baseline_pc": int(np.abs(V @ W[j]).argmax()) + 1,
                "low_end": [(rr[i], cats.get(rr[i], "?")) for i in o[:5]],
                "high_end": [(rr[i], cats.get(rr[i], "?")) for i in o[-5:][::-1]],
            })
        res["shift_pca"] = {"condition": c, "pcs": pcs,
                            "shared_shift_energy_frac": round(float((dmu ** 2).sum() / (Dd ** 2).sum(1).mean()), 3)}
        out[group] = res

    # 3. grading space: condition-mean shifts (discovery roles, discovery questions)
    G, labels = [], []
    for group in ("original", "checkable"):
        roles, conds, mu = load_centroids(args.model, group, args.rep, args.layer)
        disc = np.array([r not in held for r in roles])
        ib = conds.index("baseline")
        for c in conds:
            if c != "baseline":
                G.append((mu["mu_discovery"][disc, conds.index(c)] - mu["mu_discovery"][disc, ib]).mean(0))
                labels.append(f"{group}:{c}")
    G = np.array(G)
    U, s, Vt = np.linalg.svd(G, full_matrices=False)
    norms = np.linalg.norm(G, axis=1)
    out["grading_space"] = {
        "labels": labels,
        "sv_energy_frac": ((s ** 2) / (s ** 2).sum()).round(4).tolist(),
        "loadings_pc1_pc2_pc3": {l: (U[i, :3] * s[:3] / norms[i]).round(3).tolist() for i, l in enumerate(labels)},
        "pairwise_cos": {f"{labels[i]}|{labels[j]}": round(float(G[i] @ G[j] / norms[i] / norms[j]), 3)
                         for i in range(len(labels)) for j in range(i + 1, len(labels))},
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
