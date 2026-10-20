#!/usr/bin/env python3
"""Stage A analysis (PREREG_AB.md A.4; descriptive, discovery roles x discovery questions):
context effect at fixed text vs content effect at fixed context, per condition and group.
Projections onto the main run's frozen directions are added only when those exist (A.4,
deferred) -- pass --frozen-dir to a directory of frozen analysis vectors.

    python scripts/followup/a_analyze.py --model qwen3.6-27b --out followup_out/A
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gshift.followup.crossread import decompose  # noqa: E402

FACTORIAL = {"evaluator_main": {"auto_correct": .5, "auto_quality": .5, "human_correct": -.5, "human_quality": -.5},
             "criterion_main": {"auto_correct": .5, "human_correct": .5, "auto_quality": -.5, "human_quality": -.5},
             "betley_matched": {"auto_correct": 1, "human_quality": -1}}


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


GROUP_OF = {"auto": "original", "human": "original", "auto_correct": "checkable", "human_correct": "checkable",
            "auto_quality": "checkable", "human_quality": "checkable", "evaluator_main": "checkable",
            "criterion_main": "checkable", "betley_matched": "checkable"}
CONTRAST_OF = {"auto": "auto_minus_baseline", "human": "human_minus_baseline",
               "auto_correct": "auto_correct_minus_baseline", "human_correct": "human_correct_minus_baseline",
               "auto_quality": "auto_quality_minus_baseline", "human_quality": "human_quality_minus_baseline",
               "evaluator_main": "evaluator_main", "criterion_main": "criterion_main", "betley_matched": "betley_matched"}


def frozen_projection(vdir: Path, rep: str, L: int, c: str, M: dict) -> dict:
    """Context / content effects against the FROZEN main-run direction of the same contrast
    (fitted on discovery roles x discovery questions) and the frozen baseline persona PCs."""
    g_path = vdir / f"{GROUP_OF[c]}_{rep}_L{L}_{CONTRAST_OF[c]}.npy"
    pcs_path = vdir / f"{GROUP_OF[c]}_{rep}_L{L}_baseline_pcs.npy"
    out = {"g_file": g_path.name}
    if not g_path.exists():
        return {"skipped": f"{g_path.name} not found"}
    g = np.load(g_path).astype(np.float64)
    for k in ("CE_B", "CE_G", "TE_c", "TE_base", "Delta"):
        out[f"cos_{k}_g"] = cos(M[k], g)
        out[f"{k}_along_g_over_g"] = float(M[k] @ g / (g @ g)) if g @ g > 0 else None
    if pcs_path.exists():
        V = np.load(pcs_path).astype(np.float64)
        for k in ("CE_B", "TE_c"):
            v = M[k]
            for r in (1, 5, 20):
                rr = min(r, len(V))
                out[f"{k}_captured_by_top{r}_pcs"] = float(((V[:rr] @ v) ** 2).sum() / (v @ v)) if v @ v > 0 else None
    return out


def load(d: Path):
    cells = []
    for f in sorted(d.glob("cells_*.npz")):
        z = np.load(f)
        idx = [json.loads(s) for s in z["index"]]
        for k, row in enumerate(idx):
            cells.append((row, {rep: z[rep][k].astype(np.float64) for rep in ("pre", "resp", "reason")}))
        layers = [int(x) for x in z["layers"]]
    return cells, layers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3.6-27b")
    p.add_argument("--out", default=str(ROOT / "followup_out" / "A"))
    p.add_argument("--frozen-dir", default=None)
    args = p.parse_args()
    d = Path(args.out) / args.model
    cells, layers = load(d)
    # h[(role, qid, cond)][(text, read)] = {rep: [nL, H]}; baseline/baseline cells are shared across conds
    h = collections.defaultdict(dict)
    meta = {}
    for row, acts in cells:
        key = (row["role"], row["question_id"])
        if row["condition"] == "baseline":
            meta.setdefault(("bb",) + key, (row, acts))
            continue
        h[key + (row["condition"],)][(row["text"], row["read"])] = (row, acts)
    roles = sorted({k[0] for k in h})
    res = {"n_roles": len(roles), "roles": roles, "layers": layers, "results": {}}
    for rep in ("pre", "resp", "reason"):
        res["results"][rep] = {}
        for li, L in enumerate(layers):
            per = collections.defaultdict(lambda: collections.defaultdict(list))   # [cond][quantity] -> per-role means
            act_norms = []
            for role in roles:
                acc = collections.defaultdict(lambda: collections.defaultdict(list))
                for (r, q, c), cellset in h.items():
                    if r != role or ("bb", r, q) not in meta:
                        continue
                    bb_row, bb = meta[("bb", r, q)]
                    need = [("baseline", "graded"), ("graded", "baseline"), ("graded", "graded")]
                    if not all(k in cellset for k in need):
                        continue
                    if rep == "resp" and not (bb_row["has_answer"] and cellset[("graded", "graded")][0]["has_answer"]):
                        continue
                    if rep == "reason" and not (bb_row["n_reasoning_tokens"] and cellset[("graded", "graded")][0]["n_reasoning_tokens"]):
                        continue
                    v = {("baseline", "baseline"): bb[rep][li]}
                    v.update({k: cellset[k][1][rep][li] for k in need})
                    act_norms.append(np.linalg.norm(bb[rep][li]))
                    for name, x in decompose(v).items():
                        acc[c][name].append(x)
                for c, qd in acc.items():
                    for name, xs in qd.items():
                        per[c][name].append(np.mean(xs, 0))
            A = float(np.mean(act_norms)) if act_norms else float("nan")
            out = {"mean_activation_norm": A}
            conds = sorted(per)
            for c in conds + list(FACTORIAL):
                if c in FACTORIAL:
                    if not all(k in per for k in FACTORIAL[c]):
                        continue
                    R = min(len(per[k]["Delta"]) for k in FACTORIAL[c])
                    Q = {name: [sum(w * per[k][name][i] for k, w in FACTORIAL[c].items()) for i in range(R)]
                         for name in ("CE_B", "CE_G", "TE_base", "TE_c", "Delta", "I")}
                else:
                    Q = per[c]
                M = {name: np.mean(xs, 0) for name, xs in Q.items()}
                D = M["Delta"]
                Dn = np.linalg.norm(D)
                row = {"n_roles": len(Q["Delta"]),
                       **{f"norm_{k}_over_act": float(np.linalg.norm(M[k]) / A) for k in M},
                       "cos_CE_B_Delta": cos(M["CE_B"], D), "cos_CE_G_Delta": cos(M["CE_G"], D),
                       "cos_TE_c_Delta": cos(M["TE_c"], D), "cos_TE_base_Delta": cos(M["TE_base"], D),
                       "frac_Delta_along_CE_B": float(M["CE_B"] @ D / Dn**2) if Dn > 0 else None,
                       "frac_Delta_along_TE_c": float(M["TE_c"] @ D / Dn**2) if Dn > 0 else None,
                       "interaction_over_CE_B": float(np.linalg.norm(M["I"]) / np.linalg.norm(M["CE_B"]))
                       if np.linalg.norm(M["CE_B"]) > 0 else None}
                X = np.stack(Q["CE_B"])
                loo = [cos(X[i], np.delete(X, i, 0).mean(0)) for i in range(len(X))] if len(X) > 2 else []
                row["CE_B_mean_loo_cos_across_roles"] = float(np.nanmean(loo)) if loo else None
                if args.frozen_dir:
                    row["frozen"] = frozen_projection(Path(args.frozen_dir), rep, L, c, M)
                out[c] = row
            res["results"][rep][str(L)] = out
    (d / "analysis_A.json").write_text(json.dumps(res, indent=1))
    for rep in ("pre", "resp", "reason"):
        o = res["results"][rep].get("36", {})
        for c in ("auto", "human", "auto_correct", "human_quality", "evaluator_main", "betley_matched"):
            if c in o:
                r = o[c]
                print(f"[A] {rep} L36 {c:15s} |Δ|/act={r['norm_Delta_over_act']:.4f} |CE_B|/act={r['norm_CE_B_over_act']:.4f} "
                      f"|TE_c|/act={r['norm_TE_c_over_act']:.4f} cos(CE_B,Δ)={r['cos_CE_B_Delta']:.2f} "
                      f"cos(TE_c,Δ)={r['cos_TE_c_Delta']:.2f} |I|/|CE_B|={r['interaction_over_CE_B'] or float('nan'):.2f}")


if __name__ == "__main__":
    main()
