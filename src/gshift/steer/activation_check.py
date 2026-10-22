"""Does steering move held-out persona activations the way natural grading prompts do?

Held-out roles x validation questions x prompt variant 0, baseline condition (no grading
statement). Two readouts, both with the same HF hooks as the Q1 extractor:

  pre-response: teacher-force each prompt with and without the steering vector added at
      layer L (all positions); the displacement at DOWNSTREAM layers (> L) of the final
      prompt position is compared with the natural Q1 displacement at those layers
      (pre-response representation). At L itself the displacement is the injected vector
      by construction, so only downstream layers are informative.
  response: steered and unsteered generations (from EasySteer, benchmark "persona") are
      re-encoded (a) WITH the hook ("in context") and (b) WITHOUT it ("text only"); the
      difference of final-answer means is compared with the natural Q1 response-mean
      displacement. (b) asks whether the text the model writes under steering itself
      carries the shift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..analysis.stats import cos
from ..io import model_dir, read_jsonl
from ..templates import prompt_token_ids, split_generation
from .vectors import direction_scale


def _natural(model_key: str, rep: str, L: int, group: str = "pooled", contrast: str = "auto_minus_baseline"):
    p = model_dir(model_key) / "analysis" / "vectors" / f"{group}_{rep}_L{L}_{contrast}.npy"
    return np.load(p).astype(np.float64) if p.exists() else None


def run_activation_check(model, tokenizer, spec, prompts, manifest: dict, vectors: dict[str, np.ndarray],
                         directions: list[str], strength: float, downstream: list[int],
                         gen_dir: Path | None = None) -> dict[str, Any]:
    import torch

    from ..extract import Extractor
    from ..hooks import add_to_residual

    L = int(manifest["layer"])
    layers = sorted(set([L] + list(downstream)))
    ex = Extractor(model, tokenizer, layers, True, spec.chat_template_kwargs)
    out: dict[str, Any] = {"layer": L, "strength": strength, "downstream_layers": downstream, "pre": {}, "resp": {}}
    ids = [prompt_token_ids(tokenizer, p.messages, True, spec.chat_template_kwargs) for p in prompts]
    base = np.stack([ex.prompt_last(i) for i in ids]).astype(np.float64)          # [N, nL, H]
    nat_pre = {Lx: _natural(spec.key, "pre", Lx) for Lx in downstream}
    for d in directions:
        for sign in (1, -1):
            v = torch.tensor(direction_scale(manifest, d, sign * strength) * vectors[d], dtype=torch.float32)
            with add_to_residual(model, L, v):
                st = np.stack([ex.prompt_last(i) for i in ids]).astype(np.float64)
            delta = (st - base).mean(0)
            res = {"injected_check_cos": cos(delta[layers.index(L)], sign * vectors[d])}
            res["convergence"] = steering_convergence(spec.key, prompts, base, st, layers, downstream)
            if not res["convergence"]:
                res.pop("convergence")
            for Lx in downstream:
                if nat_pre[Lx] is not None:
                    res[f"cos_natural_L{Lx}"] = cos(delta[layers.index(Lx)], nat_pre[Lx])
                    res[f"norm_ratio_natural_L{Lx}"] = float(np.linalg.norm(delta[layers.index(Lx)])
                                                             / np.linalg.norm(nat_pre[Lx]))
            out["pre"][f"{d}_{'+' if sign > 0 else '-'}"] = res
    out["convergence_vs_random"] = convergence_vs_random(out["pre"])
    if gen_dir is not None and gen_dir.exists():
        out["resp"] = _response_check(ex, model, spec, gen_dir, manifest, vectors, strength, layers, downstream)
    return out


def steering_convergence(model_key: str, prompts, base: np.ndarray, st: np.ndarray, layers: list[int],
                         downstream: list[int], k: int = 10, n_boot: int = 500, seed: int = 0) -> dict[str, Any]:
    """Does a CONSTANT vector injected at L make personas converge further downstream?

    At L the injected vector moves every persona equally (separation ratio 1, slope 0 by
    construction). At each downstream layer and question group: separation ratio = rms spread
    of the role means with steering / without; beta = one isotropic OLS slope of each role's
    displacement on its unsteered position in the group's top-k baseline persona PCs at that
    layer (analysis/vectors/<group>_pre_L<Lx>_baseline_pcs.npy), cross-fitted over two halves
    of the questions (position from one half, displacement from the other, both ways).
    Negative beta / ratio < 1 = the network itself pulls personas together."""
    if not prompts or not all("role" in p.meta and "question_id" in p.meta for p in prompts):
        return {}
    roles = np.array([p.meta["role"] for p in prompts])
    groups = np.array([p.meta.get("group") or ("checkable" if str(p.meta["question_id"]).startswith("chk_") else "original")
                       for p in prompts])
    qids = np.array([str(p.meta["question_id"]) for p in prompts])
    vdir = model_dir(model_key) / "analysis" / "vectors"
    out: dict[str, Any] = {}
    for Lx in downstream:
        li = layers.index(Lx)
        for g in ("original", "checkable"):
            pcs_p = vdir / f"{g}_pre_L{Lx}_baseline_pcs.npy"
            sel_g = groups == g
            if not sel_g.any() or not pcs_p.exists():
                continue
            V = np.load(pcs_p).astype(np.float64)[:k]
            uq = sorted(set(qids[sel_g]))
            halves = [set(uq[0::2]), set(uq[1::2])]
            rr = sorted(set(roles[sel_g]))

            def means(A, half):
                return np.stack([A[sel_g & (roles == r) & np.isin(qids, list(half)), li].mean(0) for r in rr])

            Xb = means(base, set(uq))
            Xs = means(st, set(uq))
            spread = lambda X: float(np.sqrt(((X - X.mean(0)) ** 2).sum(1).mean()))  # noqa: E731
            xs = [means(base, halves[h]) @ V.T for h in (0, 1)]
            ds = [(means(st, halves[h]) - means(base, halves[h])) @ V.T for h in (0, 1)]

            def beta(idx):
                b = []
                for h1, h2 in ((0, 1), (1, 0)):
                    x, dlt = xs[h1][idx], ds[h2][idx]
                    xc, dc = x - x.mean(0), dlt - dlt.mean(0)
                    b.append(float((xc * dc).sum() / (xc * xc).sum()))
                return float(np.mean(b))

            # role bootstrap with a FIXED seed: every direction gets the same resamples, so
            # differences between directions (e.g. vs random) can be bootstrapped pairwise
            rng = np.random.default_rng(seed)
            boots = [beta(rng.integers(0, len(rr), len(rr))) for _ in range(n_boot)]
            out[f"{g}_L{Lx}"] = {"n_roles": len(rr), "separation_ratio": spread(Xs) / spread(Xb),
                                 "beta_crossfit": beta(np.arange(len(rr))),
                                 "beta_ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
                                 "beta_boot": [round(b, 5) for b in boots]}
    return out


def convergence_vs_random(pre: dict[str, Any]) -> dict[str, Any]:
    """beta(direction) - beta(random, same sign) per cell, with the paired role-bootstrap CI."""
    out: dict[str, Any] = {}
    for key, v in pre.items():
        d, sign = key.rsplit("_", 1)
        if d.startswith("random") or "convergence" not in v:
            continue
        for rkey, rv in pre.items():
            rd, rsign = rkey.rsplit("_", 1)
            if not rd.startswith("random") or rsign != sign or "convergence" not in rv:
                continue
            for cell, e in v["convergence"].items():
                r = rv["convergence"].get(cell)
                if not r or "beta_boot" not in e or "beta_boot" not in r:
                    continue
                diff = np.array(e["beta_boot"]) - np.array(r["beta_boot"])
                out.setdefault(key, {}).setdefault(rd, {})[cell] = {
                    "beta_minus_random": e["beta_crossfit"] - r["beta_crossfit"],
                    "ci95": np.percentile(diff, [2.5, 97.5]).tolist()}
    return out


def _resp_mean(ex, rec, hook_ctx=None) -> np.ndarray | None:
    sp = split_generation(rec["gen_ids"], ex.tok, ex.info)
    if not sp.answer:
        return None
    ids = rec["prompt_ids"] + rec["gen_ids"][: sp.answer[-1] + 1]
    P = len(rec["prompt_ids"])
    if hook_ctx is None:
        hs = ex.forward_capture(ids)
    else:
        with hook_ctx():
            hs = ex.forward_capture(ids)
    return np.stack([hs[L][[P + i for i in sp.answer]].double().mean(0).cpu().numpy() for L in ex.layers])


def _response_check(ex, model, spec, gen_dir: Path, manifest, vectors, strength, layers, downstream) -> dict:
    import torch

    from ..hooks import add_to_residual

    L = int(manifest["layer"])
    unsteered = {r["id"]: r for r in read_jsonl(gen_dir / "none.jsonl")}
    nat = {Lx: _natural(spec.key, "resp", Lx) for Lx in layers}
    base_means = {i: _resp_mean(ex, r) for i, r in unsteered.items()}
    out = {}
    for f in sorted(gen_dir.glob("*.jsonl")):
        cond = f.stem
        if cond == "none" or cond.startswith("natural_"):
            continue
        recs = read_jsonl(f)
        if not recs or abs(abs(recs[0]["signed_strength"]) - strength) > 1e-9:
            continue
        d, signed = recs[0]["direction"], recs[0]["signed_strength"]
        v = torch.tensor(direction_scale(manifest, d, signed) * vectors[d], dtype=torch.float32)
        ctx_d, txt_d = [], []
        for r in recs:
            b = base_means.get(r["id"])
            if b is None:
                continue
            m_txt = _resp_mean(ex, r)
            m_ctx = _resp_mean(ex, r, lambda: add_to_residual(model, L, v))
            if m_txt is None:
                continue
            txt_d.append(m_txt - b)
            ctx_d.append(m_ctx - b)
        if not txt_d:
            continue
        txt, ctx = np.mean(txt_d, 0), np.mean(ctx_d, 0)
        res = {"n": len(txt_d)}
        for Lx in layers:
            if nat.get(Lx) is not None:
                res[f"text_only_cos_natural_L{Lx}"] = cos(txt[layers.index(Lx)], nat[Lx])
                if Lx in downstream:
                    res[f"in_context_cos_natural_L{Lx}"] = cos(ctx[layers.index(Lx)], nat[Lx])
        out[cond] = res
    return out
