"""Context probes: last-prompt-token residuals under extra context conditions (forward pass
only, no generation). Design in configs/probes.yaml; GPU job scripts/15_context_probes.py;
CPU report gshift.analysis.probes_analysis / scripts/16_probe_report.py.

Position. The captured position is the LAST prompt token: the "\\n" after the generation
prompt's <think> (thinking ON, the model's chat_template_kwargs) -- the same token as Q1's
`pre` (gshift.extract.Extractor.record takes h[P - 1] of the identical rendering).

Batching. Prompts are RIGHT-padded and run without an attention mask. Every operation in
the Qwen3.5 decoder is causal (full attention with a causal mask; Gated-DeltaNet's causal
conv1d and recurrence), so the residual at position i depends only on tokens 0..i: the
residual at each row's last REAL position (len - 1) is unaffected by the pad tokens after
it, and no recurrent state ever consumes a pad token before a real one. Positions are the
default arange, identical to an unbatched pass. Checked on a tiny Qwen3.5 model in
tests/test_probes.py and, on the real model, by `self_check` (unbatched re-run) before any
file is written.

Early stop. Only layers <= max(capture_layers) are needed: the forward pass is aborted
after the last captured block input (a quarter of the compute for layers [24, 36, 48] of
64).

Layout (one file per condition / pair side; resumable per file):
    outputs/<model>/probes/persona/<condition>.npy        [n, n_layers, H] float16
    outputs/<model>/probes/persona/<condition>.index.jsonl rows (role, question_id, group, ...)
    outputs/<model>/probes/persona/index.jsonl            all conditions (file, row, ...)
    outputs/<model>/probes/control/<pair>_<A|B>.npy (+ .index.jsonl, index.jsonl)
    outputs/<model>/probes/{persona,control}/meta.json, outputs/<model>/probes/summary.json
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from . import CONFIG_DIR, REPO_ROOT
from .config import heldout_roles, load_questions, load_yaml, role_categories
from .design import build_messages, role_instruction
from .io import model_dir, read_jsonl, write_json

QUESTION_PLACEHOLDER = "{question}"


# ---------------------------------------------------------------------------
# design (pure)
# ---------------------------------------------------------------------------

def load_probe_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(path or CONFIG_DIR / "probes.yaml")


def probes_dir(model_key: str) -> Path:
    return model_dir(model_key) / "probes"


@dataclass(frozen=True)
class ProbePrompt:
    key: str                         # unique within its file
    probe_set: str                   # persona | control
    name: str                        # file name: condition (persona) or <pair>_<side> (control)
    messages: tuple[dict[str, str], ...]
    user_content: str
    role: str | None = None
    category: str | None = None
    question_id: str | None = None
    group: str | None = None
    family: str | None = None
    condition: str | None = None
    pair: str | None = None
    side: str | None = None
    task_id: str | None = None
    task_category: str | None = None
    statement: str | None = None     # prefix sentence (None: baseline / template conditions / no prefix)

    def meta(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("messages")
        return {k: v for k, v in d.items() if v is not None}


def discovery_roles(exp: dict[str, Any]) -> list[str]:
    """Prompted roles minus the held-out split, in config order (80 for the frozen design)."""
    held = set(heldout_roles(exp))
    return [r for r in role_categories(exp) if r not in held]


def probe_questions(exp: dict[str, Any], which: str = "discovery") -> list[dict[str, Any]]:
    rows = load_questions(exp)
    return rows if which == "all" else [r for r in rows if r["split"] == which]


def grading_statement(exp: dict[str, Any], wording: str, key: str) -> str:
    """Q1 statement text for wording id (w1..) and 'evaluator|criterion' key."""
    for ws in exp["wording_templates"].values():
        if wording in ws:
            return ws[wording][key]
    raise KeyError(f"unknown wording {wording!r}")


def condition_user(exp: dict[str, Any], spec: dict[str, Any], question: str, group: str) -> tuple[str, str | None]:
    """(user message, prefix statement or None) for one persona-probe condition. Prefix
    conditions use Q1's own build_messages / placement, so auto_w1 is byte-identical to a
    Q1 w1 prompt and baseline to a Q1 baseline prompt."""
    kind = spec["kind"]
    placement = exp.get("placement", "user_prefix")
    if kind == "baseline":
        return question, None
    if kind == "template":
        tmpl = spec["template"]
        if tmpl.count(QUESTION_PLACEHOLDER) != 1:
            raise ValueError(f"template must contain {QUESTION_PLACEHOLDER} exactly once: {tmpl!r}")
        return tmpl.replace(QUESTION_PLACEHOLDER, question), None     # literal, never .format
    if kind == "grading":
        text = grading_statement(exp, spec["wording"], spec["by_group"][group])
    elif kind == "prefix":
        text = spec["text"]
    else:
        raise ValueError(f"unknown condition kind {kind!r}")
    _, user = build_messages(None, question, text, placement)
    return user, text


def persona_prompts(exp: dict[str, Any], pcfg: dict[str, Any], roles: list[str] | None = None,
                    questions: list[dict] | None = None) -> dict[str, list[ProbePrompt]]:
    """{condition: prompts} in (role, question) order: roles in config order, questions in
    manifest order (original then checkable as in the manifest)."""
    pp = pcfg["persona_probe"]
    if roles is None:
        if pp.get("roles", "discovery") != "discovery":
            raise ValueError("persona_probe.roles: only 'discovery' is supported")
        roles = discovery_roles(exp)
    questions = probe_questions(exp, pp.get("questions", "discovery")) if questions is None else questions
    cats = role_categories(exp)
    pv = int(pp.get("prompt_variant", 0))
    out: dict[str, list[ProbePrompt]] = {}
    for cond, spec in pp["conditions"].items():
        rows = []
        for role in roles:
            system = role_instruction(role, pv) or None
            for q in questions:
                user, stmt = condition_user(exp, spec, q["text"], q["group"])
                msgs = ([{"role": "system", "content": system}] if system else []) + [
                    {"role": "user", "content": user}]
                rows.append(ProbePrompt(key=f"{role}|{q['id']}", probe_set="persona", name=cond,
                                        messages=tuple(msgs), user_content=user, role=role,
                                        category=cats.get(role), question_id=q["id"], group=q["group"],
                                        family=q["family"], condition=cond, statement=stmt))
        out[cond] = rows
    return out


def load_control_tasks(pcfg: dict[str, Any], verify: bool = True) -> list[dict[str, Any]]:
    cp = pcfg["control_pairs"]
    path = Path(cp["tasks"])
    path = path if path.is_absolute() else REPO_ROOT / path
    raw = path.read_bytes()
    if verify and cp.get("tasks_sha256") and hashlib.sha256(raw).hexdigest() != cp["tasks_sha256"]:
        raise RuntimeError(f"{path} changed since the probe design was frozen (sha256 mismatch)")
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()]


def control_prompts(pcfg: dict[str, Any], pairs: list[str] | None = None,
                    tasks: list[dict] | None = None) -> dict[str, list[ProbePrompt]]:
    """{<pair>_<side>: prompts}: no system message, statement as a user prefix (null = none)."""
    cp = pcfg["control_pairs"]
    tasks = load_control_tasks(pcfg) if tasks is None else tasks
    out: dict[str, list[ProbePrompt]] = {}
    for pair, sides in cp["pairs"].items():
        if pairs is not None and pair not in pairs:
            continue
        for side in ("A", "B"):
            stmt = sides.get(side)
            rows = []
            for t in tasks:
                _, user = build_messages(None, t["text"], stmt, "user_prefix")
                rows.append(ProbePrompt(key=t["id"], probe_set="control", name=f"{pair}_{side}",
                                        messages=({"role": "user", "content": user},), user_content=user,
                                        pair=pair, side=side, task_id=t["id"], task_category=t.get("category"),
                                        statement=stmt))
            out[f"{pair}_{side}"] = rows
    return out


def smoke_subsets(exp: dict[str, Any], pcfg: dict[str, Any]) -> dict[str, Any]:
    """Roles / questions / pairs / tasks of the --smoke run (first n of each)."""
    sm = pcfg.get("smoke", {})
    qs = probe_questions(exp, pcfg["persona_probe"].get("questions", "discovery"))
    nq = int(sm.get("n_questions_per_group", 1))
    picked = []
    for g in dict.fromkeys(q["group"] for q in qs):
        picked += [q for q in qs if q["group"] == g][:nq]
    return {"roles": discovery_roles(exp)[: int(sm.get("n_roles", 2))], "questions": picked,
            "pairs": list(pcfg["control_pairs"]["pairs"])[: int(sm.get("n_pairs", 2))],
            "tasks": load_control_tasks(pcfg)[: int(sm.get("n_tasks", 4))]}


# ---------------------------------------------------------------------------
# batching (pure)
# ---------------------------------------------------------------------------

def make_batches(lengths: Sequence[int], max_batch_tokens: int, max_batch_size: int) -> list[list[int]]:
    """Index batches, longest first, so that len(batch) * max_len(batch) <= max_batch_tokens
    (a single over-long prompt still gets its own batch)."""
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        width = lengths[cur[0]] if cur else lengths[i]          # longest first: first is widest
        if cur and ((len(cur) + 1) * width > max_batch_tokens or len(cur) >= max_batch_size):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


def pad_right(seqs: Sequence[Sequence[int]], pad_id: int) -> tuple[np.ndarray, np.ndarray]:
    """[B, T] right-padded ids and each row's last NON-PAD position (len - 1). The pad id
    never matters: it only ever sits after a row's last real token (causal model)."""
    if not seqs or any(len(s) == 0 for s in seqs):
        raise ValueError("empty prompt")
    T = max(len(s) for s in seqs)
    x = np.full((len(seqs), T), int(pad_id), dtype=np.int64)
    for b, s in enumerate(seqs):
        x[b, : len(s)] = s
    return x, np.array([len(s) - 1 for s in seqs], dtype=np.int64)


# ---------------------------------------------------------------------------
# capture (torch)
# ---------------------------------------------------------------------------

class _StopForward(Exception):
    pass


def capture_last(model, batch_ids, last_idx, layers: list[int], early_stop: bool = True) -> dict[int, Any]:
    """{L: [B, H] float32 residual at row b's position last_idx[b]} for one padded batch.
    Same convention as gshift.hooks.capture_residual (L < n: input of block L; L == n:
    output of the last block)."""
    import torch

    from .hooks import _first_tensor, find_text_model

    tm = find_text_model(model)
    blocks = tm.layers
    n = len(blocks)
    layers = sorted(set(int(L) for L in layers))
    for L in layers:
        if not 0 <= L <= n:
            raise ValueError(f"layer {L} out of range 0..{n}")
    B = batch_ids.shape[0]
    store: dict[int, Any] = {}
    stop_at = layers[-1] if (early_stop and layers[-1] < n) else None
    handles = []

    def take(h):
        rows = torch.arange(B, device=h.device)
        return h[rows, last_idx.to(h.device)].float()

    for L in layers:
        if L < n:
            def pre_hook(module, args, kwargs, L=L):
                store[L] = take(_first_tensor(args[0] if args else kwargs.get("hidden_states")))
                if L == stop_at:
                    raise _StopForward
            handles.append(blocks[L].register_forward_pre_hook(pre_hook, with_kwargs=True))
        else:
            def post_hook(module, args, output, L=L):
                store[L] = take(_first_tensor(output))
            handles.append(blocks[n - 1].register_forward_hook(post_hook))
    try:
        with torch.inference_mode():
            try:
                tm(input_ids=batch_ids, use_cache=False)
            except _StopForward:
                pass
    finally:
        for h in handles:
            h.remove()
    missing = [L for L in layers if L not in store]
    if missing:
        raise RuntimeError(f"layers {missing} were not captured")
    return store


def capture_prompts(model, token_lists: Sequence[Sequence[int]], layers: list[int], pad_id: int,
                    max_batch_tokens: int = 32768, max_batch_size: int = 64, early_stop: bool = True,
                    log: Callable[[str], None] | None = None) -> np.ndarray:
    """[N, n_layers, H] float32 last-prompt-token residuals, rows in input order."""
    import torch

    from .extract import input_device

    layers = sorted(set(int(L) for L in layers))
    dev = input_device(model)
    out: np.ndarray | None = None
    batches = make_batches([len(t) for t in token_lists], max_batch_tokens, max_batch_size)
    t0 = time.time()
    for k, idx in enumerate(batches):
        x, last = pad_right([token_lists[i] for i in idx], pad_id)
        st = capture_last(model, torch.from_numpy(x).to(dev), torch.from_numpy(last), layers, early_stop)
        arr = np.stack([st[L].cpu().numpy() for L in layers], axis=1)          # [B, nL, H]
        if out is None:
            out = np.zeros((len(token_lists),) + arr.shape[1:], dtype=np.float32)
        out[idx] = arr
        if log and (k + 1) % 20 == 0:
            log(f"  batch {k + 1}/{len(batches)} ({time.time() - t0:.0f}s)")
    assert out is not None
    return out


def self_check(model, token_lists: Sequence[Sequence[int]], layers: list[int], pad_id: int,
               n: int = 4, **kw) -> dict[str, Any]:
    """Batched (right-padded, mixed lengths) vs one-prompt-at-a-time residuals on the real
    model: min cosine and max relative L2 error over rows x layers."""
    if not token_lists:
        return {"n": 0}
    # the n longest and n shortest prompts, batched together (maximal padding)
    order = sorted(range(len(token_lists)), key=lambda i: len(token_lists[i]))
    pick = sorted(set(order[:n] + order[-n:]))
    seqs = [token_lists[i] for i in pick]
    batched = capture_prompts(model, seqs, layers, pad_id, max_batch_tokens=10**9, max_batch_size=len(seqs), **kw)
    single = np.concatenate([capture_prompts(model, [s], layers, pad_id, **kw) for s in seqs])
    num = (batched * single).sum(-1)
    den = np.linalg.norm(batched, axis=-1) * np.linalg.norm(single, axis=-1)
    cosv = num / np.where(den > 0, den, 1)
    rel = np.linalg.norm(batched - single, axis=-1) / np.maximum(np.linalg.norm(single, axis=-1), 1e-12)
    return {"n": len(seqs), "lengths": [len(s) for s in seqs], "min_cos": float(cosv.min()),
            "max_rel_err": float(rel.max()), "mean_rel_err": float(rel.mean())}


# ---------------------------------------------------------------------------
# run + checkpoint
# ---------------------------------------------------------------------------

def token_sha(ids: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest()[:16]


def tokenize_prompts(tok, prompts: Sequence[ProbePrompt], enable_thinking: bool = True,
                     template_kwargs: dict | None = None) -> list[list[int]]:
    from .templates import prompt_token_ids

    return [prompt_token_ids(tok, p.messages, enable_thinking, template_kwargs) for p in prompts]


def _file_done(out_dir: Path, name: str, rows: list[dict]) -> bool:
    idx = out_dir / f"{name}.index.jsonl"
    if not idx.exists() or not (out_dir / f"{name}.npy").exists():
        return False
    have = read_jsonl(idx)
    return [(r["key"], r["prompt_sha"]) for r in have] == [(r["key"], r["prompt_sha"]) for r in rows]


def save_file(out_dir: Path, name: str, acts: np.ndarray, rows: list[dict], save_dtype: str = "float16") -> None:
    """Array first, index last (atomic rename): an index file implies a complete array."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / f"{name}.tmp.npy"
    np.save(tmp, acts.astype(np.dtype(save_dtype)))
    tmp.replace(out_dir / f"{name}.npy")
    ti = out_dir / f"{name}.index.jsonl.tmp"
    ti.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    ti.replace(out_dir / f"{name}.index.jsonl")


def run_probe_set(model, tok, prompt_sets: dict[str, list[ProbePrompt]], out_dir: Path, layers: list[int],
                  enable_thinking: bool = True, template_kwargs: dict | None = None,
                  save_dtype: str = "float16", max_batch_tokens: int = 32768, max_batch_size: int = 64,
                  check: dict | None = None, meta: dict | None = None,
                  log: Callable[[str], None] = print) -> dict[str, Any]:
    """Capture every file of a probe set (resumable per file); returns timing/token stats.
    `check` = {"n": .., "min_cos": ..}: batched-vs-unbatched self check on the first file
    that needs computing, before anything is written (raises if it fails)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    if isinstance(pad_id, (list, tuple)):
        pad_id = pad_id[0]
    write_json(out_dir / "meta.json", {"layers": sorted(set(layers)), "save_dtype": save_dtype,
                                       "position": "last prompt token (after the generation prompt)",
                                       "padding": "right, no attention mask (causal model)",
                                       "enable_thinking": enable_thinking, "template_kwargs": template_kwargs or {},
                                       **(meta or {})})
    stats: dict[str, Any] = {"files": {}, "self_check": None}
    all_rows: list[dict] = []
    for name, prompts in prompt_sets.items():
        ids = tokenize_prompts(tok, prompts, enable_thinking, template_kwargs)
        rows = [{"file": name, "row": i, **p.meta(), "n_prompt_tokens": len(t), "prompt_sha": token_sha(t)}
                for i, (p, t) in enumerate(zip(prompts, ids))]
        all_rows += rows
        n_tok = int(sum(len(t) for t in ids))
        if _file_done(out_dir, name, rows):
            stats["files"][name] = {"n": len(rows), "tokens": n_tok, "skipped": True}
            continue
        if check and stats["self_check"] is None:
            sc = self_check(model, ids, layers, pad_id, n=int(check.get("n", 4)))
            stats["self_check"] = sc
            log(f"[probes] self check (batched vs unbatched): {json.dumps(sc)}")
            if sc["n"] and sc["min_cos"] < float(check.get("min_cos", 0.999)):
                raise RuntimeError(f"batched capture disagrees with unbatched passes: {sc}")
        t0 = time.time()
        acts = capture_prompts(model, ids, layers, pad_id, max_batch_tokens, max_batch_size, log=log)
        dt = time.time() - t0
        save_file(out_dir, name, acts, rows, save_dtype)
        stats["files"][name] = {"n": len(rows), "tokens": n_tok, "seconds": round(dt, 1),
                                "mean_norm": [float(x) for x in np.linalg.norm(acts, axis=-1).mean(0)]}
        log(f"[probes] {out_dir.name}/{name}: {len(rows)} prompts, {n_tok} tokens, {dt:.0f}s")
    ti = out_dir / "index.jsonl.tmp"
    ti.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))
    ti.replace(out_dir / "index.jsonl")
    return stats


def quick_summary(out_dir: Path, names: list[str], ref: str, layers: list[int], layer: int) -> dict[str, Any]:
    """Per file at `layer`: mean activation norm and ||mean(file) - mean(ref)|| (rows matched
    by key). Small, for the job log; the real analysis is scripts/16_probe_report.py."""
    if layer not in layers or not (out_dir / f"{ref}.npy").exists():
        return {}
    li = sorted(set(layers)).index(layer)
    base = np.load(out_dir / f"{ref}.npy", mmap_mode="r")[:, li].astype(np.float64)
    res = {}
    for name in names:
        if not (out_dir / f"{name}.npy").exists():
            continue
        a = np.load(out_dir / f"{name}.npy", mmap_mode="r")[:, li].astype(np.float64)
        res[name] = {"mean_norm": float(np.linalg.norm(a, axis=1).mean()),
                     "mean_shift_norm_vs_" + ref: float(np.linalg.norm(a.mean(0) - base.mean(0)))
                     if a.shape == base.shape else None}
    return res


# ---------------------------------------------------------------------------
# helpers shared with the analysis
# ---------------------------------------------------------------------------

def family_halves(families: Sequence[str]) -> list[tuple[frozenset, frozenset]]:
    """All splits of a group's families into two equal halves (one per complementary pair)."""
    fams = sorted(set(families))
    k = len(fams) // 2
    if k == 0:
        return []
    out: list[tuple[frozenset, frozenset]] = []
    for c in itertools.combinations(fams, k):
        a, b = frozenset(c), frozenset(fams) - frozenset(c)
        if len(a) == len(b) and not any(b == x for x, _ in out):
            out.append((a, b))
    return out
