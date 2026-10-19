"""Residual-stream capture with one, explicit layer convention.

LAYER CONVENTION (shared with Betley et al., public-steering-vectors vectors/0007):
    layer L, 0 <= L < n_layers : residual stream at the INPUT of decoder block L
                                 == HF output_hidden_states[L] == output of block L-1
    layer L == n_layers        : output of the last block (before the final norm)
So "layer 36" here is the same tensor as Betley's layer 36 and as the Assistant Axis
code's "layer 35" (assistant-axis hooks block OUTPUTS). Captured with a
forward_pre_hook on block L (L < n_layers) or a forward hook on the last block.
Steering "at layer L" adds the vector to that same tensor, i.e. to the OUTPUT of block
L-1 (exactly what public-steering-vectors/pod does: block = layer - 1).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch


def find_text_model(model: torch.nn.Module, n_layers: int | None = None) -> torch.nn.Module:
    """Return the text decoder module (owns `embed_tokens` and `layers`), refusing to guess.

    Matches the unique module with a ModuleList `layers` and an `embed_tokens`
    (the text decoder; a vision tower has blocks but no token embedding).
    """
    matches = []
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and hasattr(module, "embed_tokens"):
            if n_layers is None or len(layers) == n_layers:
                matches.append((name or "<root>", module))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one decoder stack (n_layers={n_layers}), found "
            f"{[(n, len(m.layers)) for n, m in matches]}"
        )
    return matches[0][1]


def find_decoder_layers(model: torch.nn.Module, n_layers: int | None = None) -> torch.nn.ModuleList:
    """The decoder block list of the text model (see find_text_model)."""
    return find_text_model(model, n_layers).layers


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return x[0]
    raise RuntimeError(f"unsupported decoder block input/output type: {type(x)}")


@contextmanager
def capture_residual(
    model: torch.nn.Module, layers: list[int], n_layers: int | None = None
) -> Iterator[dict[int, torch.Tensor]]:
    """Context manager filling {layer: hidden [batch, seq, hidden]} on each forward."""
    blocks = find_decoder_layers(model, n_layers)
    n = len(blocks)
    store: dict[int, torch.Tensor] = {}
    handles = []
    for L in layers:
        if not 0 <= L <= n:
            raise ValueError(f"layer {L} out of range 0..{n}")
        if L < n:
            def pre_hook(module, args, kwargs, L=L):
                hs = args[0] if args else kwargs.get("hidden_states")
                store[L] = _first_tensor(hs).detach()
            handles.append(blocks[L].register_forward_pre_hook(pre_hook, with_kwargs=True))
        else:
            def post_hook(module, args, output, L=L):
                store[L] = _first_tensor(output).detach()
            handles.append(blocks[n - 1].register_forward_hook(post_hook))
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@contextmanager
def add_to_residual(
    model: torch.nn.Module,
    layer: int,
    vector: torch.Tensor,
    positions: str = "all",
    n_layers: int | None = None,
) -> Iterator[None]:
    """HF-side steering, same location as EasySteer/public-steering-vectors.

    Adds `vector` (already scaled) to the residual stream at `layer` in the convention
    above, i.e. to the OUTPUT of block layer-1, at every position. Used for the
    activation check (does steering reproduce the natural shift downstream?) and as a
    reference implementation for testing the EasySteer path.
    """
    if positions != "all":
        raise NotImplementedError("only all-position steering is used in this project")
    blocks = find_decoder_layers(model, n_layers)
    if not 1 <= layer <= len(blocks):
        raise ValueError(f"steering layer {layer} must be in 1..{len(blocks)}")
    target = blocks[layer - 1]

    def hook(module, args, output):
        if isinstance(output, torch.Tensor):
            return output + vector.to(output.dtype).to(output.device)
        if isinstance(output, tuple):
            return (output[0] + vector.to(output[0].dtype).to(output[0].device),) + tuple(output[1:])
        raise RuntimeError(f"unsupported block output {type(output)}")

    h = target.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()
