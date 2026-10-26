#!/usr/bin/env python3
"""Does grading TRANSLATE personas, CONTRACT them toward a point, or ROTATE them? Where is the point?

Uses saved Q1 persona centroids (open-ended questions, baseline vs graded conditions) for Qwen3.6/3.8 and
OLMo DPO/final. In the top-k baseline persona PCs (discovery-question centroids), with x_r a persona's
baseline position and d_r its displacement under a condition, fit on DISCOVERY questions and score on the
independent VALIDATION-question centroids:

  T  translation            d_r = c
  C  contraction to a point  d_r = -kappa * x_r + c        (fixed point p = c / kappa; kappa = share of
                                                             the distance to p closed)
  D  per-axis contraction    d_r = -diag(kappa) x_r + c
  F  general linear          d_r = A x_r + c   (A = symmetric part [stretch/shrink] + skew part [rotation])

Reports held-out R^2 of the validation displacements (around 0, and beyond the mean shift), the fixed
point's location (distance to the default Assistant, nearest personas, norm vs persona spread), the
rotation share of A, and the share of the displacement that lies OUTSIDE the persona subspace.

    python scripts/psm/fixed_point.py   -> docs/psm/fixed_point/fixed_point.json + printout
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
Q = ROOT.parent / "grader-shifts-persona-space" / "outputs"
SRC = {"qwen3.6-27b": Q / "qwen3.6-27b" / "analysis" / "vectors",
       "qwen3.8-27b": Q / "qwen3.8-27b" / "analysis" / "vectors",
       "olmo3-32b-dpo": ROOT / "outputs" / "_hf_q1" / "olmo3-32b-dpo" / "analysis" / "vectors",
       "olmo31-32b-final": ROOT / "outputs" / "_hf_q1" / "olmo31-32b-final" / "analysis" / "vectors"}
K = 10


def load(m, rep, L):
    z = np.load(SRC[m] / f"original_{rep}_L{L}_condition_centroids.npz", allow_pickle=True)
    return ([str(r) for r in z["roles"]], [str(c) for c in z["conds"]],
            z["mu_discovery"].astype(np.float64), z["mu_validation"].astype(np.float64),
            z["mu_default_discovery"].astype(np.float64), z["mu_default_validation"].astype(np.float64))


def r2(pred, y, around_mean=False):
    if around_mean:
        y0 = y - y.mean(0)
        pred0 = pred - y.mean(0)
        return 1 - ((y - pred) ** 2).sum() / (y0 ** 2).sum()
    return 1 - ((y - pred) ** 2).sum() / (y ** 2).sum()


def fit_models(x, d):
    """x, d: [R, k] -> fitted predictors (functions of x) and parameters."""
    R, k = x.shape
    out = {}
    c = d.mean(0)
    out["T"] = (lambda X, c=c: np.broadcast_to(c, X.shape), {})
    # C: d = -kappa x + c  (shared kappa): least squares over all coordinates
    xm, dm = x - x.mean(0), d - d.mean(0)
    kappa = -float((xm * dm).sum() / (xm * xm).sum())
    cC = d.mean(0) + kappa * x.mean(0)
    out["C"] = (lambda X, kp=kappa, cc=cC: -kp * X + cc, {"kappa": kappa, "fixed_point": (cC / kappa) if abs(kappa) > 1e-9 else None})
    # D: per-axis kappa
    kd = -np.array([(xm[:, j] * dm[:, j]).sum() / (xm[:, j] ** 2).sum() for j in range(k)])
    cD = d.mean(0) + kd * x.mean(0)
    out["D"] = (lambda X, kk=kd, cc=cD: -kk * X + cc, {"kappa_per_axis": kd})
    # F: general linear, ridge-stabilised (R >= 50 personas, k = 10)
    Xa = np.hstack([x, np.ones((R, 1))])
    lam = 1e-3 * np.trace(Xa.T @ Xa) / (k + 1)
    W = np.linalg.solve(Xa.T @ Xa + lam * np.eye(k + 1), Xa.T @ d)
    A, cF = W[:k].T, W[k]
    S, Kskew = (A + A.T) / 2, (A - A.T) / 2
    out["F"] = (lambda X, A=A, cc=cF: X @ A.T + cc,
                {"rotation_share": float(np.linalg.norm(Kskew) / np.linalg.norm(A)),
                 "sym_eigs": np.linalg.eigvalsh(S).tolist()})
    return out


def analyse(m, rep, L, cond):
    roles, conds, mu_d, mu_v, md_d, md_v = load(m, rep, L)
    if cond not in conds:
        return None
    b, g = conds.index("baseline"), conds.index(cond)
    Xd, Yd, Xv, Yv = mu_d[:, b], mu_d[:, g], mu_v[:, b], mu_v[:, g]
    m0 = Xd.mean(0)
    U, S, Vt = np.linalg.svd(Xd - m0, full_matrices=False)
    V = Vt[:K]
    proj = lambda Z: (Z - m0) @ V.T  # noqa: E731
    xd, dd = proj(Xd), proj(Yd) - proj(Xd)
    xv, dv = proj(Xv), proj(Yv) - proj(Xv)
    # share of the (mean) displacement inside the top-K persona subspace
    full_shift = (Yd - Xd).mean(0)
    inside = float(np.linalg.norm(V @ full_shift) ** 2 / np.linalg.norm(full_shift) ** 2)
    per_role_full = Yd - Xd
    inside_roles = float(np.mean(np.linalg.norm(per_role_full @ V.T, axis=1) ** 2 / np.linalg.norm(per_role_full, axis=1) ** 2))
    models = fit_models(xd, dd)
    res = {"roles": len(roles), "k": K, "share_of_mean_shift_in_persona_subspace": inside,
           "share_of_per_role_shift_in_persona_subspace": inside_roles}
    # ceiling: how well do discovery displacements predict validation displacements, role by role?
    res["ceiling_R2"] = float(r2(dd, dv))
    res["ceiling_R2_beyond_mean"] = float(r2(dd - dd.mean(0) + dv.mean(0), dv, around_mean=True))
    for name, (f, par) in models.items():
        pred = f(xv)
        res[name] = {"R2": float(r2(pred, dv)), "R2_beyond_mean": float(r2(pred, dv, around_mean=True))}
        if name == "C":
            kp, p = par["kappa"], par["fixed_point"]
            res[name]["kappa"] = kp
            if p is not None:
                # fixed point in full space (inside the persona subspace), vs default Assistant and personas
                P = m0 + p @ V
                spread = float(np.linalg.norm(Xd - m0, axis=1).mean())
                dfl = (md_d[b] + md_v[b]) / 2
                d_def = float(np.linalg.norm(proj(dfl[None])[0] - p))
                dists = np.linalg.norm(xd - p, axis=1)
                near = [roles[i] for i in np.argsort(dists)[:5]]
                res[name].update({"fixed_point_norm_over_spread": float(np.linalg.norm(p) / spread),
                                  "fixed_point_to_default_over_spread": d_def / spread,
                                  "default_norm_over_spread": float(np.linalg.norm(proj(dfl[None])[0]) / spread),
                                  "nearest_personas_to_fixed_point": near,
                                  "cos_fixedpoint_dir_with_default_dir": float(p @ proj(dfl[None])[0] /
                                                                               (np.linalg.norm(p) * np.linalg.norm(proj(dfl[None])[0]) + 1e-12))})
        if name == "F":
            res[name].update({"rotation_share": par["rotation_share"],
                              "sym_eigs_min_max": [min(par["sym_eigs"]), max(par["sym_eigs"])]})
        if name == "D":
            res[name]["kappa_per_axis"] = [float(x) for x in par["kappa_per_axis"]]
    return res


def main():
    out = {}
    for m in SRC:
        for rep in ("pre", "resp"):
            for cond in ("auto", "human", "neutral"):
                r = analyse(m, rep, 36, cond)
                if r:
                    out.setdefault(m, {}).setdefault(rep, {})[cond] = r
    od = ROOT / "docs" / "psm" / "fixed_point"
    od.mkdir(parents=True, exist_ok=True)
    (od / "fixed_point.json").write_text(json.dumps(out, indent=1))
    for m, d in out.items():
        for rep, dc in d.items():
            for cond, r in dc.items():
                C = r["C"]
                print(f"{m:17} {rep:4} {cond:7} | held-out R2 (beyond mean shift): T {r['T']['R2_beyond_mean']:+.2f}  "
                      f"C {r['C']['R2_beyond_mean']:+.2f}  D {r['D']['R2_beyond_mean']:+.2f}  F {r['F']['R2_beyond_mean']:+.2f}  "
                      f"(ceiling {r['ceiling_R2_beyond_mean']:+.2f}) | kappa {C['kappa']:.2f} | "
                      f"fixed pt |p|/spread {C.get('fixed_point_norm_over_spread', float('nan')):.2f}, "
                      f"to default {C.get('fixed_point_to_default_over_spread', float('nan')):.2f} "
                      f"(default |.|/spread {C.get('default_norm_over_spread', float('nan')):.2f}), "
                      f"cos(p, default) {C.get('cos_fixedpoint_dir_with_default_dir', float('nan')):+.2f} | "
                      f"near {C.get('nearest_personas_to_fixed_point', [])[:3]} | rotation share {r['F']['rotation_share']:.2f} | "
                      f"shift in persona subspace: mean {r['share_of_mean_shift_in_persona_subspace']:.2f}, per-role {r['share_of_per_role_shift_in_persona_subspace']:.2f}")


if __name__ == "__main__":
    main()
