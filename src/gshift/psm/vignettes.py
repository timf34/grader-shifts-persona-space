"""Track B trait direction from third-person vignettes (accurate vs knowingly misrepresenting).

Each vignette is read as a user message "Here is a short scene.\\n\\n<setup> <ending>" in the
model's own chat template. The representation is the mean residual over the ENDING's tokens
(located as the span where the two renderings of the same vignette differ from the setup-only
rendering: common prefix / common suffix of the token ids), at each layer. Direction at layer L =
mean over TRAIN vignettes of h(misrepresent) - h(accurate) (paired difference of means).
Held-out check: fraction of held-out pairs whose paired difference projects positively, and the
standardised separation of the two endings along the direction (a representational positive
control; it does not show causal use).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import REPO_ROOT

INTRO = "Here is a short scene.\n\n"


def load_vignettes(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    p = p if p.is_absolute() else REPO_ROOT / p
    return [json.loads(l) for l in open(p) if l.strip()]


def _msgs(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


def ending_span(full: list[int], setup_only: list[int]) -> tuple[int, int]:
    """[start, end) of the tokens that differ: common prefix / suffix with the setup-only ids."""
    c = 0
    while c < min(len(full), len(setup_only)) and full[c] == setup_only[c]:
        c += 1
    s = 0
    while s < min(len(full), len(setup_only)) - c and full[-1 - s] == setup_only[-1 - s]:
        s += 1
    start, end = c, len(full) - s
    if end <= start:
        raise ValueError("empty ending span")
    return start, end


def vignette_features(ex, tokenizer, vignettes: list[dict], template_kwargs: dict | None = None
                      ) -> dict[str, np.ndarray]:
    """-> {"accurate": [N, nL, H], "misrepresent": [N, nL, H], "last_accurate", "last_misrepresent"}
    (ending-mean and last-prompt-token residuals) for ex.layers."""
    from ..templates import prompt_token_ids

    out: dict[str, list] = {"accurate": [], "misrepresent": [], "last_accurate": [], "last_misrepresent": []}
    for v in vignettes:
        base = prompt_token_ids(tokenizer, _msgs(INTRO + v["setup"]), True, template_kwargs)
        for side in ("accurate", "misrepresent"):
            ids = prompt_token_ids(tokenizer, _msgs(INTRO + v["setup"] + " " + v[side]), True, template_kwargs)
            a, b = ending_span(ids, base)
            hs = ex.forward_capture(ids)
            out[side].append(np.stack([hs[L][a:b].double().mean(0).cpu().numpy() for L in ex.layers]))
            out["last_" + side].append(np.stack([hs[L][-1].double().cpu().numpy() for L in ex.layers]))
    return {k: np.stack(v).astype(np.float32) for k, v in out.items()}


def trait_directions(feats: dict[str, np.ndarray], splits: list[str], layers: list[int],
                     rep: str = "ending") -> dict[str, Any]:
    """Unit directions per layer from TRAIN pairs + held-out validation stats."""
    acc = feats["accurate" if rep == "ending" else "last_accurate"].astype(np.float64)
    mis = feats["misrepresent" if rep == "ending" else "last_misrepresent"].astype(np.float64)
    tr = np.array([s == "train" for s in splits])
    ho = ~tr
    dirs, stats = {}, {}
    for j, L in enumerate(layers):
        d = (mis[tr, j] - acc[tr, j]).mean(0)
        u = d / np.linalg.norm(d)
        dirs[L] = u.astype(np.float32)
        pd = (mis[ho, j] - acc[ho, j]) @ u
        pa, pm = acc[ho, j] @ u, mis[ho, j] @ u
        pooled_sd = np.sqrt(0.5 * (pa.var(ddof=1) + pm.var(ddof=1)))
        # split-half stability of the direction within TRAIN
        idx = np.flatnonzero(tr)
        h1, h2 = idx[::2], idx[1::2]
        d1, d2 = (mis[h1, j] - acc[h1, j]).mean(0), (mis[h2, j] - acc[h2, j]).mean(0)
        stats[L] = {"n_train": int(tr.sum()), "n_heldout": int(ho.sum()),
                    "heldout_pair_sign_acc": float((pd > 0).mean()),
                    "heldout_d": float((pm.mean() - pa.mean()) / pooled_sd) if pooled_sd > 0 else None,
                    "train_split_half_cos": float(d1 @ d2 / np.linalg.norm(d1) / np.linalg.norm(d2)),
                    "raw_norm": float(np.linalg.norm(d))}
    return {"dirs": dirs, "stats": stats}
