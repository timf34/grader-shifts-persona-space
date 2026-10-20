"""Stage D (implemented, not run tonight): donor-matched coordinate replacement, adapted from
value transplant (Jiang & Roger) to TWO CONTEXTS OF ONE MODEL.

The host generates under its context; the donor reads the SAME task and the same
host-generated continuation under a different context and never generates. Before each new
host token, the host's residual at OUR layer L (input of block L, last position) is edited

    h'_H = h_H + lambda * U U^T (h_D - h_H)          U: [k, H] orthonormal rows

so at lambda = 1 the host's coordinates inside span(U) are replaced by the donor's
(donor-matched coordinate replacement; NOT a recreation of the donor's full computation).

Controls (PREREG §C/D): constant push (a fixed vector of matched per-token magnitude),
shuffled donor (donor coordinates taken from another trajectory / position), norm-matched
random direction, same-context donor (implementation check: must be a no-op), and
intervention-and-release (edit the first T tokens, then continue unedited, either with the
cache built under intervention or with a cache recomputed from the text alone).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass
class TransplantConfig:
    layer: int                       # our convention: input of block `layer`
    U: np.ndarray                    # [k, H] orthonormal rows
    lam: float = 1.0
    mode: str = "live"               # live | constant | shuffled | none
    constant: np.ndarray | None = None          # [H] for mode=constant
    shuffled_coords: list[np.ndarray] | None = None   # per-step [k] donor coords for mode=shuffled
    release_after: int | None = None            # stop editing after this many generated tokens
    recompute_cache_on_release: bool = False
    log: list[dict] = field(default_factory=list)


def _orthonormal(U: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(np.asarray(U, np.float64).T)
    return q.T


def generate_with_transplant(model, host_ids: list[int], donor_ids: list[int], cfg: TransplantConfig,
                             max_new_tokens: int, sample: Callable | None = None, stop_ids: set[int] | None = None
                             ) -> tuple[list[int], list[dict]]:
    """Greedy (or `sample(logits) -> token`) generation with per-token editing.

    host_ids / donor_ids: the two rendered prompts (same task, different contexts). The donor
    reads every host-generated token (never edited, never generating). Full recompute per
    step (batch 1, no KV cache: safer with the hybrid Gated-DeltaNet stack), but with KV-cache
    SEMANTICS: each step's edit is stored by absolute position and re-applied on every later
    recompute, and the host coordinate is read with the earlier edits in place -- exactly what
    an edited KV cache would carry forward. After `release_after` tokens no new edits are
    made; earlier ones are kept (the "intervention cache") unless recompute_cache_on_release,
    in which case they are dropped (a cache recomputed from the text alone).
    Returns (generated ids, per-step log of host/donor coordinates)."""
    import torch

    from ..hooks import capture_residual, find_decoder_layers

    dev = next(model.parameters()).device
    blocks = find_decoder_layers(model)
    target = blocks[cfg.layer - 1]                      # our layer L = output of block L-1
    U = torch.tensor(_orthonormal(cfg.U), dtype=torch.float32, device=dev)          # [k, H]
    edits: dict[int, "torch.Tensor"] = {}               # absolute position -> added vector
    gen: list[int] = []
    log: list[dict] = []
    stop_ids = stop_ids or set()
    P = len(host_ids)

    def donor_coord(ids):
        with torch.no_grad(), capture_residual(model, [cfg.layer]) as st:
            model(input_ids=torch.tensor([ids], device=dev))
        return U @ st[cfg.layer][0, -1].float()

    for t in range(max_new_tokens):
        released = cfg.release_after is not None and t >= cfg.release_after
        if released and cfg.recompute_cache_on_release:
            edits.clear()
        new_edit = cfg.mode != "none" and not released
        pos = P + t - 1                                 # position whose state predicts token t
        c_d = None
        if new_edit and cfg.mode == "live":
            c_d = donor_coord(donor_ids + gen)
        elif new_edit and cfg.mode == "shuffled":
            c_d = torch.tensor(cfg.shuffled_coords[t % len(cfg.shuffled_coords)], dtype=torch.float32, device=dev)
        state = {}

        def hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            out = out.clone()
            for q, d in edits.items():
                out[:, q, :] += d.to(out.dtype)
            if new_edit:
                h = out[0, pos].float()
                c_h = U @ h
                if cfg.mode in ("live", "shuffled"):
                    d = cfg.lam * (U.T @ (c_d - c_h))
                else:                                   # constant push
                    d = torch.tensor(cfg.constant, dtype=torch.float32, device=dev)
                out[:, pos, :] += d.to(out.dtype)
                state.update(edit=d, c_h=c_h)
            return (out,) + tuple(output[1:]) if isinstance(output, tuple) else out

        handle = target.register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([host_ids + gen], device=dev)).logits[0, -1].float()
        finally:
            handle.remove()
        if new_edit:
            edits[pos] = state["edit"]
            log.append({"t": t, "host": state["c_h"].cpu().numpy(),
                        "donor": None if c_d is None else c_d.cpu().numpy()})
        nxt = int(sample(logits) if sample else logits.argmax())
        gen.append(nxt)
        if nxt in stop_ids:
            break
    return gen, log


def matched_constant(log: list[dict], U: np.ndarray, lam: float) -> np.ndarray:
    """Constant-push control: the mean live edit vector of a reference run (same per-token
    magnitude on average, no moment-to-moment information)."""
    Uo = _orthonormal(U)
    d = np.mean([lam * (e["donor"] - e["host"]) for e in log if e["donor"] is not None], axis=0)
    return Uo.T @ d


def random_like(U: np.ndarray, seed: int = 0) -> np.ndarray:
    """Norm-matched random-direction control: a random orthonormal basis of the same rank."""
    rng = np.random.default_rng(seed)
    return _orthonormal(rng.normal(size=U.shape))
