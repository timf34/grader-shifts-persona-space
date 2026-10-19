"""Figures for the Q1 analysis (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CAT_COLORS = {"analytical_helping": "#1f77b4", "relational": "#2ca02c", "competitive": "#d62728",
              "creative": "#9467bd", "nonhuman": "#8c564b", "default": "#000000"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_arrows(path: Path, P_base: np.ndarray, P_cond: np.ndarray, roles: list[str], cats: list[str],
                title: str, dims=(0, 1), P_def: tuple[np.ndarray, np.ndarray] | None = None,
                heldout: set[str] | None = None):
    """Each persona's movement baseline -> condition in the FIXED baseline-PCA frame."""
    plt = _plt()
    i, j = dims
    # Right panel magnifies arrows so their DIRECTIONS are visible; the shift can be small
    # relative to persona spread in these two PCs without being absent (it may live
    # outside the baseline PC plane -- see captured_frac in summary.json).
    arrows = P_cond[:, [i, j]] - P_base[:, [i, j]]
    span = np.ptp(P_base[:, [i, j]], axis=0).max() or 1.0
    med = np.median(np.linalg.norm(arrows, axis=1)) or 1.0
    mag = max(1.0, 0.08 * span / med)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, scale in zip(axes, (1.0, mag)):
        for k, r in enumerate(roles):
            c = CAT_COLORS.get(cats[k], "grey")
            ax.scatter(P_base[k, i], P_base[k, j], color=c, s=28, zorder=3,
                       marker="s" if heldout and r in heldout else "o")
            end = P_base[k, [i, j]] + scale * arrows[k]
            ax.annotate("", xy=(end[0], end[1]), xytext=(P_base[k, i], P_base[k, j]),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
            ax.text(P_base[k, i], P_base[k, j], " " + r, fontsize=7, color=c)
        if P_def is not None:
            b, c2 = P_def
            ax.scatter(b[i], b[j], marker="*", s=160, color="black", zorder=4)
            end = b[[i, j]] + scale * (c2[[i, j]] - b[[i, j]])
            ax.annotate("", xy=(end[0], end[1]), xytext=(b[i], b[j]), arrowprops=dict(arrowstyle="->", color="black"))
            ax.text(b[i], b[j], "  default", fontsize=8)
        ax.set_xlabel(f"baseline PC{i + 1}")
        ax.set_ylabel(f"baseline PC{j + 1}")
        ax.set_title(title + ("" if scale == 1.0 else f"  [arrows x{scale:.0f}]"), fontsize=9)
    for cat, col in CAT_COLORS.items():
        axes[0].scatter([], [], color=col, label=cat)
    axes[0].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_cos_heatmap(path: Path, C: np.ndarray, roles: list[str], title: str):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(roles)), roles, rotation=90, fontsize=6)
    ax.set_yticks(range(len(roles)), roles, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_role(path: Path, per_role: list[dict], title: str):
    plt = _plt()
    roles = [d["role"] for d in per_role]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(roles, [d["norm"] for d in per_role])
    axes[0].set_title("||Delta_r||", fontsize=9)
    axes[1].bar(roles, [d["loo_cos"] for d in per_role])
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("cos(Delta_r, mean of other roles)", fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", rotation=90, labelsize=6)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_spectrum(path: Path, spec: dict, title: str):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(spec["energy_frac"], "o-", label="D (uncentred)")
    ax.plot(spec["residual_energy_frac"], "s-", label="D - shared")
    ax.set_xlabel("component")
    ax.set_ylabel("energy fraction")
    ax.legend(fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_layer_block(fig_dir: Path, rep: str, L: int, res: dict, cells, mu, mu_def, space, gs, conds,
                     heldout: list[str] | None = None):
    from .space import project

    fig_dir.mkdir(parents=True, exist_ok=True)
    pca = space["_pca"]
    cats = [cells.categories.get(r, "?") for r in cells.roles]
    ib = conds.index("baseline")
    Pb = project(pca, mu[:, ib], 3)
    for c in conds:
        if c == "baseline":
            continue
        Pc = project(pca, mu[:, conds.index(c)], 3)
        pdef = None
        if mu_def is not None and np.all(np.isfinite(mu_def)):
            pdef = (project(pca, mu_def[ib][None], 3)[0], project(pca, mu_def[conds.index(c)][None], 3)[0])
        for dims in [d for d in ((0, 1), (0, 2)) if max(d) < Pb.shape[1]]:
            plot_arrows(fig_dir / f"arrows_{rep}_L{L}_{c}_pc{dims[0] + 1}{dims[1] + 1}.png", Pb, Pc, cells.roles,
                        cats, f"{rep} L{L}: baseline -> {c} (PCA fit on discovery roles; squares = held-out)",
                        dims, pdef, set(heldout or []))
    for name in gs:
        blk = res[name]
        roles = [d["role"] for d in blk["per_role"]]
        # rebuild the cosine matrix from the saved displacement file next to the vectors
        vec = fig_dir.parent / "vectors" / f"{rep}_L{L}_{name}_role_displacements.npz"
        if vec.exists():
            plot_cos_heatmap(fig_dir / f"cos_{rep}_L{L}_{name}.png", np.load(vec)["pairwise_cos"], roles,
                             f"{rep} L{L} {name}: cos(Delta_r, Delta_s), discovery roles")
        plot_per_role(fig_dir / f"per_role_{rep}_L{L}_{name}.png", blk["per_role"], f"{rep} L{L} {name}")
        plot_spectrum(fig_dir / f"spectrum_{rep}_L{L}_{name}.png", blk["spectrum"], f"{rep} L{L} {name}")


def plot_sweep(fig_dir: Path, sweep, primary_layer: int):
    plt = _plt()
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics = [m for m in ["g_norm_over_act_norm", "mean_loo_cos", "split_half_cos", "heldout_role_cos",
                           "heldout_question_cos", "heldout_both_cos", "sv1_energy_frac", "crossfit_dz"]
               if m in sweep.columns] + [c for c in sweep.columns if c.startswith("cos_g_")]
    for (group, rep), sub in sweep.groupby(["group", "rep"]):
        fig, axes = plt.subplots(2, (len(metrics) + 1) // 2, figsize=(4 * ((len(metrics) + 1) // 2), 7))
        for ax, m in zip(axes.flat, metrics):
            for name, s in sub.groupby("contrast"):
                if s[m].isna().all():
                    continue
                ax.plot(s["layer"], s[m].astype(float), "o-", label=name)
                if m == "mean_loo_cos" and "loo_cos_null_q95" in s:
                    ax.plot(s["layer"], s["loo_cos_null_q95"].astype(float), ":", color="grey")
            ax.axvline(primary_layer, color="k", lw=0.5, ls="--")
            ax.set_title(m, fontsize=9)
            ax.set_xlabel("layer (block input)")
        axes.flat[0].legend(fontsize=6)
        fig.suptitle(f"layer sweep: {group} / {rep} (discovery estimates)", fontsize=10)
        fig.tight_layout()
        fig.savefig(fig_dir / f"sweep_{group}_{rep}.png", dpi=150)
        plt.close(fig)
