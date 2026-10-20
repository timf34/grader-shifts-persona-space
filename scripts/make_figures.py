#!/usr/bin/env python3
"""Curated Q1 result figures for one model, from outputs/<model>/analysis/ (summary.json +
vectors/). Writes outputs/<model>/analysis/figures_report/*.png and headline.json.

    python scripts/make_figures.py --model qwen3.8-27b

Palette: the dataviz reference categorical slots (validated: lightness, chroma, CVD,
normal-vision floor; three slots sit below 3:1 on the surface, so every figure carries
a legend / direct labels and categories also differ by marker shape).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gshift.config import heldout_roles, load_experiment, role_categories  # noqa: E402
from gshift.io import outputs_root  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3dd", "#fcfcfb"
CATS = ["analytical_helping", "relational", "competitive", "creative", "nonhuman"]
CAT_LABEL = {"analytical_helping": "analytical / helping", "relational": "relational", "competitive": "competitive",
             "creative": "creative", "nonhuman": "nonhuman"}
CAT_MARK = {"analytical_helping": "o", "relational": "s", "competitive": "^", "creative": "D", "nonhuman": "v"}
REP_LABEL = {"pre": "last prompt token (pre)", "resp": "final-answer tokens (resp)", "reason": "reasoning tokens (reason)"}
KEY = [("original", "auto_minus_baseline", "open-ended: automated − baseline"),
       ("original", "human_minus_baseline", "open-ended: human − baseline"),
       ("original", "auto_minus_human", "open-ended: automated − human"),
       ("checkable", "evaluator_main", "checkable: evaluator main effect"),
       ("checkable", "criterion_main", "checkable: criterion main effect"),
       ("checkable", "betley_matched", "checkable: auto+correct − human+quality (Betley)")]


def style():
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": GRID,
        "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6})


def save(fig, path: Path, title: str, note: str | None = None):
    # reserve bands for the title and the note so neither collides with titles / tick labels
    fig.tight_layout(rect=(0, 0.07 if note else 0, 1, 0.93))
    fig.suptitle(title, x=0.01, y=0.99, ha="left", fontsize=13, fontweight="bold")
    if note:
        fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=8.5, color=INK2, wrap=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def blk(S, rep, L, group, name):
    return S["results"].get(rep, {}).get(str(L), {}).get(group, {}).get(name)


def fig_sweep(S, reps, out):
    layers = [int(x) for x in S["layers"]]
    fig, axes = plt.subplots(2, len(reps), figsize=(4.6 * len(reps), 7.2), sharex=True, squeeze=False)
    for j, rep in enumerate(reps):
        for i, (metric, ylab) in enumerate((("g_norm_over_act_norm", "|g| / mean activation norm"),
                                            ("mean_loo_cos", "mean leave-one-role-out cosine"))):
            ax = axes[i, j]
            for c, (grp, name, lab) in enumerate(KEY):
                ys = []
                for L in layers:
                    b = blk(S, rep, L, grp, name)
                    ys.append(np.nan if b is None else (b.get(metric) if metric in b else b["observed"].get(metric)))
                ax.plot(layers, ys, "-o", color=SERIES[c], label=lab, markersize=5)
            ax.axvline(S["primary_layer"], color=INK2, lw=1, ls=":")
            if i == 0:
                ax.set_title(REP_LABEL[rep])
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("layer (input of block L)")
                ax.set_ylim(-0.1, 1.02)
    axes[1, 0].legend(loc="lower left", fontsize=8)     # the LOO panels are empty below ~0.6
    save(fig, out / "fig1_layer_sweep.png", f"{S['model_key']}: size and consistency of the grading shift across layers",
         "Top: norm of the shared shift g (mean over 80 discovery roles of graded − baseline centroids) relative to the "
         "typical activation norm. Bottom: how well each role's shift aligns with the mean of the OTHER roles' shifts "
         "(1 = identical direction). Dotted line: pre-registered layer 36.")


def fig_shared(S, reps, exp, out):
    L = S["primary_layer"]
    cats = role_categories(exp)
    names = KEY
    fig, axes = plt.subplots(1, len(reps), figsize=(5.2 * len(reps), 5.6), sharey=True, squeeze=False)
    for j, rep in enumerate(reps):
        ax = axes[0, j]
        rng = np.random.default_rng(0)
        for i, (grp, name, lab) in enumerate(names):
            b = blk(S, rep, L, grp, name)
            if not b:
                continue
            for cat in CATS:
                vals = [p["loo_cos"] for p in b["per_role"] if cats.get(p["role"]) == cat]
                ax.scatter(np.array(vals), i + rng.uniform(-0.28, 0.28, len(vals)), s=16, marker=CAT_MARK[cat],
                           color=SERIES[CATS.index(cat)], alpha=0.85, linewidths=0,
                           label=CAT_LABEL[cat] if (i == 0 and j == 0) else None)
            q = b["null"].get("mean_loo_cos", {}).get("null_q95")
            if q is not None:
                ax.plot([q, q], [i - 0.38, i + 0.38], color=INK, lw=1.4)
            m = b["observed"]["mean_loo_cos"]
            ax.text(1.04, i, f"{m:.2f}", va="center", fontsize=8.5, color=INK, transform=ax.get_yaxis_transform())
        ax.set_yticks(range(len(names)))
        if j == 0:
            ax.set_yticklabels([n[2] for n in names], fontsize=8.5)
        else:
            ax.tick_params(labelleft=False)
        ax.set_xlim(-0.6, 1.0)
        ax.set_xlabel("role's cosine with the other roles' mean shift")
        ax.set_title(REP_LABEL[rep])
        ax.invert_yaxis()
    axes[0, 0].legend(loc="lower left", fontsize=8, ncol=1)
    save(fig, out / "fig2_shared_shift_per_role.png", f"{S['model_key']}: is the grading shift shared across personas? (layer {L})",
         "One dot per discovery role. Black tick: 95th percentile of the paired sign-flip null for the mean. "
         "Number at right: mean over roles.")


def fig_persona(S, rep, exp, out, group="original", cond="auto"):
    L = S["primary_layer"]
    vec = outputs_root() / S["model_key"] / "analysis" / "vectors"
    z = np.load(vec / f"{group}_{rep}_L{L}_condition_centroids.npz")
    V = np.load(vec / f"{group}_{rep}_L{L}_baseline_pcs.npy").astype(np.float64)
    c0 = np.load(vec / f"{group}_{rep}_L{L}_baseline_pca_center.npy").astype(np.float64)
    roles, conds = list(z["roles"]), list(z["conds"])
    mu = z["mu_discovery"].astype(np.float64)
    ib, ic = conds.index("baseline"), conds.index(cond)
    Pb, Pc = (mu[:, ib] - c0) @ V[:2].T, (mu[:, ic] - c0) @ V[:2].T
    cats = role_categories(exp)
    held = set(heldout_roles(exp))
    evr = S["results"][rep][str(L)][group]["persona_space"]["pca_evr"]
    fig, ax = plt.subplots(figsize=(8.6, 7.4))
    for r, role in enumerate(roles):
        cat = cats.get(role)
        if cat is None:
            continue
        col = SERIES[CATS.index(cat)]
        ax.annotate("", xy=Pc[r], xytext=Pb[r], arrowprops=dict(arrowstyle="-|>", color=col, lw=1.1, alpha=0.75,
                                                                shrinkA=0, shrinkB=0, mutation_scale=8))
        ax.scatter(*Pb[r], marker=CAT_MARK[cat], s=26, facecolors="none" if role in held else col,
                   edgecolors=col, linewidths=1.2, zorder=3)
    if "mu_default_discovery" in z:
        d = z["mu_default_discovery"].astype(np.float64)
        db, dc = (d[ib] - c0) @ V[:2].T, (d[ic] - c0) @ V[:2].T
        ax.annotate("", xy=dc, xytext=db, arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=10))
        ax.scatter(*db, marker="*", s=180, color=INK, zorder=4)
        ax.text(db[0], db[1], "  default Assistant", fontsize=9, color=INK, va="center")
    for cat in CATS:
        ax.scatter([], [], marker=CAT_MARK[cat], color=SERIES[CATS.index(cat)], label=CAT_LABEL[cat])
    ax.scatter([], [], marker="o", facecolors="none", edgecolors=INK2, label="held-out role (hollow)")
    ax.legend(loc="best", fontsize=8.5)
    ax.set_xlabel(f"PC1 of baseline personas ({100 * evr[0]:.0f}% of variance)")
    ax.set_ylabel(f"PC2 ({100 * evr[1]:.0f}%)")
    label = {"auto": "“…evaluated automatically”", "human": "“…evaluated by a human”"}.get(cond, cond)
    save(fig, out / f"fig3_persona_space_{group}_{rep}_{cond}.png",
         f"{S['model_key']}: personas move from baseline to {label} ({REP_LABEL[rep]}, layer {L})",
         "PCA fit on discovery-role baseline centroids only; every role and the condition are projected into that fixed "
         "basis. Marker = baseline position, arrow head = graded position. Arrows that look short can still be large "
         "outside these two PCs.")


def fig_geometry(S, reps, out):
    L = S["primary_layer"]
    rows = []
    for rep in reps:
        for grp, name, lab in KEY:
            b = blk(S, rep, L, grp, name)
            g = (b or {}).get("geometry", {})
            if "observed" in g:
                rows.append((rep, lab, g))
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.42 * len(rows) + 1.8), sharey=True)
    ylab = [f"{lab}  [{rep}]" for rep, lab, _ in rows]
    for ax, (key, ref, title) in zip(axes, (("separation_ratio", 1.0, "separation ratio (graded / baseline)"),
                                            ("beta", 0.0, "contraction β (shift vs baseline position)"),
                                            ("shared_energy_frac_cf", 1.0, "shared-translation energy fraction"))):
        for i, (rep, lab, g) in enumerate(rows):
            o = g["observed"].get(key)
            ci = g["bootstrap_ci"].get(key, {})
            col = SERIES[[r for r in reps].index(rep)]
            if ci:
                ax.plot([ci["lo"], ci["hi"]], [i, i], color=col, lw=2.2, solid_capstyle="round")
            if o is not None:
                ax.scatter(o, i, color=col, s=36, zorder=3, edgecolors=SURF, linewidths=1.5)
        ax.axvline(ref, color=INK2, lw=1, ls=":")
        ax.set_title(title)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels(ylab, fontsize=8)
    axes[0].invert_yaxis()
    for rep in reps:
        axes[0].scatter([], [], color=SERIES[reps.index(rep)], label=REP_LABEL[rep])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=len(reps), fontsize=8.5,
               bbox_to_anchor=(0.99, 1.0))
    verdicts = [f"{lab} [{rep}]: " + ", ".join(f"{k.split('_', 1)[0]} {v}" for k, v in g["verdict"].items())
                for rep, lab, g in rows[:2]]
    save(fig, out / "fig4_geometry_H1_H2_H3.png", f"{S['model_key']}: translation, convergence or position-dependent shift? (layer {L})",
         "Cross-fitted over whole question-family halves; bars = role-bootstrap 95% CIs. Translation (H1): separation "
         "ratio ≈ 1 and β ≈ 0. Convergence (H2): ratio < 1 and β < 0. " + " | ".join(verdicts))


def fig_factorial(S, reps, out):
    L = S["primary_layer"]
    names = [("evaluator_main", "evaluator (auto − human)"), ("criterion_main", "criterion (correct − quality)"),
             ("evaluator_x_criterion", "evaluator × criterion")]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    w = 0.8 / len(reps)
    for j, rep in enumerate(reps):
        for i, (name, _) in enumerate(names):
            b = blk(S, rep, L, "checkable", name)
            if not b:
                continue
            act = b["g_norm"] / max(b["g_norm_over_act_norm"], 1e-12) if b.get("g_norm_over_act_norm") else None
            x = i + (j - (len(reps) - 1) / 2) * w
            ax.bar(x, b["g_norm_over_act_norm"], width=w * 0.92, color=SERIES[j], label=REP_LABEL[rep] if i == 0 else None)
            q = b["null"].get("g_norm", {}).get("null_q95")
            if q is not None and act:
                ax.plot([x - w * 0.42, x + w * 0.42], [q / act] * 2, color=INK, lw=1.6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n[1] for n in names])
    ax.set_ylabel("|g| / mean activation norm")
    ax.legend(fontsize=8.5)
    save(fig, out / "fig5_evaluator_vs_criterion.png", f"{S['model_key']}: who grades vs what is graded (checkable tasks, layer {L})",
         "Black ticks: 95th percentile of |g| under the paired sign-flip null (noise floor).")


def fig_generalisation(S, reps, out):
    L = S["primary_layer"]
    views = [("heldout_roles_x_discovery_questions", "held-out roles"),
             ("discovery_roles_x_validation_questions", "held-out questions"),
             ("heldout_roles_x_validation_questions", "both held out")]
    fig, axes = plt.subplots(1, len(reps), figsize=(5.4 * len(reps), 4.8), sharey=True, squeeze=False)
    for j, rep in enumerate(reps):
        ax = axes[0, j]
        for c, (grp, name, lab) in enumerate(KEY[:4]):
            b = blk(S, rep, L, grp, name)
            if not b:
                continue
            ys = [b["generalisation"].get(v, {}).get("cos_g", np.nan) for v, _ in views]
            vw = S["results"][rep][str(L)].get("validation_wordings", {})
            ys.append(vw.get(f"{grp}|{name}|combined", {}).get("cos_g_wording_g_discovery", np.nan))
            ax.plot(range(len(ys)), ys, "-o", color=SERIES[c], label=lab if j == 0 else None)
        ax.set_xticks(range(len(views) + 1))
        ax.set_xticklabels([v[1] for v in views] + ["unseen wordings"], rotation=15)
        ax.set_ylim(-0.1, 1.02)
        ax.set_title(REP_LABEL[rep])
        if j == 0:
            ax.set_ylabel("cosine with the discovery direction g")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    save(fig, out / "fig6_generalisation.png", f"{S['model_key']}: does the discovered direction generalise? (layer {L})",
         "Direction estimated on discovery roles × discovery questions × discovery wordings, compared with the same "
         "contrast estimated on data it never saw.")


def fig_adherence(S, out):
    a = S["adherence"]["by_group_condition"]
    order = [("original", ["baseline", "auto", "human"]),
             ("checkable", ["baseline", "auto_correct", "human_correct", "auto_quality", "human_quality"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 5]}, sharey=True)
    for ax, (g, conds) in zip(axes, order):
        ys = [a.get(f"{g}|{c}", {}).get("mean") for c in conds]
        fr = [a.get(f"{g}|{c}", {}).get("frac_fully_roleplaying") for c in conds]
        ax.bar(range(len(conds)), ys, color=[INK2 if c == "baseline" else SERIES[0] for c in conds], width=0.62)
        for i, (y, f) in enumerate(zip(ys, fr)):
            if y is not None:
                ax.text(i, y + 0.04, f"{y:.2f}\n({100 * f:.0f}% full)", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([c.replace("_", "\n") for c in conds])
        ax.set_title("open-ended questions" if g == "original" else "checkable tasks")
        ax.set_ylim(0, 3.4)
    axes[0].set_ylabel("mean role-adherence score (0–3)")
    save(fig, out / "fig7_role_adherence.png", f"{S['model_key']}: do personas stay in character when told they are graded?",
         "Assistant Axis rubric judged by gpt-4.1-mini on the final answer only (question without the grading sentence). "
         "Gray = baseline. '% full' = share scored 3 (fully role-playing).")


def fig_assistant(S, reps, out):
    L = S["primary_layer"]
    fig, axes = plt.subplots(1, len(reps), figsize=(5.0 * len(reps), 4.4), sharey=True, squeeze=False)
    ks = [1, 2, 3, 5, 10, 20, 40, 79]
    for j, rep in enumerate(reps):
        ax = axes[0, j]
        sp = S["results"][rep][str(L)]["original"]["persona_space"]
        for c, name in enumerate(["auto_minus_baseline", "human_minus_baseline", "auto_minus_human"]):
            cap = sp.get(name, {}).get("captured_frac", {})
            ax.plot(ks, [cap.get(str(k), np.nan) for k in ks], "-o", color=SERIES[c], label=KEY[c][2] if j == 0 else None)
        rnd = sp.get("auto_minus_baseline", {}).get("random_direction_expectation", {})
        ax.plot(ks, [rnd.get(str(k), np.nan) for k in ks], "--", color=INK2, lw=1.2, label="random direction" if j == 0 else None)
        ac = sp.get("assistant_contrast", {})
        ttl = REP_LABEL[rep]
        if ac:
            aligned = "not aligned" if "NOT" in str(ac["pc1_label"]) else "aligned"
            ttl += f"\nPC1 vs Assistant contrast: |cos| = {ac['abs_cos_with_pc1']:.2f} ({aligned})"
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.set_xlabel("number of baseline persona PCs")
        if j == 0:
            ax.set_ylabel("fraction of ‖g‖² inside the PC subspace")
    axes[0, 0].legend(fontsize=8)
    save(fig, out / "fig8_inside_persona_space.png", f"{S['model_key']}: how much of the grading shift lies in persona space? (layer {L})",
         "Captured fraction of the shared shift by the top-k baseline persona PCs (discovery roles), vs the expectation "
         "for a random direction. Outside this sampled subspace ≠ outside all persona representations.")


def headline(S, reps):
    L = str(S["primary_layer"])
    h = {"model": S["model_key"], "layer": int(L), "n_records": S["n_records"], "contrasts": {}}
    for rep in reps:
        for grp, name, lab in KEY:
            b = blk(S, rep, L, grp, name)
            if not b:
                continue
            g = b.get("geometry", {})
            h["contrasts"][f"{rep}|{grp}|{name}"] = {
                "label": lab, "g_norm_over_act": b["g_norm_over_act_norm"], "mean_loo_cos": b["observed"]["mean_loo_cos"],
                "loo_p": b["null"].get("mean_loo_cos", {}).get("p_greater"), "g_norm_p": b["null"].get("g_norm", {}).get("p_greater"),
                "split_half_cos": b["split_half"]["mean_cos"],
                "heldout_role_cos": b["generalisation"].get("heldout_roles_x_discovery_questions", {}).get("cos_g"),
                "heldout_question_cos": b["generalisation"].get("discovery_roles_x_validation_questions", {}).get("cos_g"),
                "cos_betley_repro": b.get("cos_g_betley_repro"), "sv1_energy_frac": b["observed"]["sv1_energy_frac"],
                "separation_ratio": g.get("observed", {}).get("separation_ratio"), "beta": g.get("observed", {}).get("beta"),
                "verdict": g.get("verdict")}
    h["adherence"] = S["adherence"]["by_group_condition"]
    return h


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    args = p.parse_args()
    style()
    exp = load_experiment()
    adir = outputs_root() / args.model / "analysis"
    S = json.loads((adir / "summary.json").read_text())
    out = adir / "figures_report"
    out.mkdir(parents=True, exist_ok=True)
    reps = [r for r in ("pre", "resp", "reason") if r in S["results"]]
    fig_sweep(S, reps, out)
    fig_shared(S, reps, exp, out)
    for rep in reps:
        for grp, cond in (("original", "auto"), ("original", "human"), ("checkable", "auto_correct")):
            try:
                fig_persona(S, rep, exp, out, grp, cond)
            except FileNotFoundError as e:
                print("skip persona plot:", e)
    fig_geometry(S, reps, out)
    fig_factorial(S, reps, out)
    fig_generalisation(S, reps, out)
    fig_adherence(S, out)
    fig_assistant(S, reps, out)
    (out / "headline.json").write_text(json.dumps(headline(S, reps), indent=1, default=float))
    print("wrote", out / "headline.json")


if __name__ == "__main__":
    main()
