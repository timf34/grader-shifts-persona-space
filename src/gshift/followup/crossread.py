"""Stage A cross-reading (PREREG_AB.md A): the same generated tokens read under the baseline
context and under a grading context, for matched baseline / graded generations of the main
run. Pure bookkeeping here; the forward passes use gshift.extract.Extractor unchanged."""

from __future__ import annotations

import collections
from typing import Any

import numpy as np

from ..config import heldout_roles
from ..design import build_requests

ROLE_PRIORITY_SEED = 20261022
PER_CATEGORY = 4
MAIN_CONDS = ("baseline", "auto", "human", "auto_correct", "human_correct", "auto_quality", "human_quality")


def role_priority(exp: dict) -> dict[str, list[str]]:
    """Per category: seeded permutation of its DISCOVERY roles (config order)."""
    held = set(heldout_roles(exp))
    rng = np.random.default_rng(ROLE_PRIORITY_SEED)
    out = {}
    for cat, roles in exp["roles"].items():
        disc = [r for r in roles if r not in held]
        out[cat] = [disc[i] for i in rng.permutation(len(disc))]
    return out


def select_roles(exp: dict, complete: set[str]) -> tuple[list[str], dict[str, int]]:
    """First PER_CATEGORY complete roles per category in priority order; shortfalls recorded."""
    sel, short = [], {}
    for cat, order in role_priority(exp).items():
        got = [r for r in order if r in complete][:PER_CATEGORY]
        sel += got
        if len(got) < PER_CATEGORY:
            short[cat] = PER_CATEGORY - len(got)
    return sel, short


def designed_ids(exp: dict, role: str) -> dict[str, set[str]]:
    by = collections.defaultdict(set)
    for r in build_requests(exp, "main", [role], include_default=False):
        by[r.condition].add(r.id)
    return by


def validate_file(rows: list[dict], want: set[str], config_id: str) -> list[str]:
    """A.1 (i)-(iii): reasons the file fails, empty if it passes."""
    why = []
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)):
        why.append("duplicate ids")
    if set(ids) != want:
        why.append(f"ids differ from design (missing {len(want - set(ids))}, extra {len(set(ids) - want)})")
    cids = {r.get("config_id") for r in rows}
    if cids != {config_id}:
        why.append(f"config ids {sorted(map(str, cids))} != {config_id}")
    if any(not r.get("prompt_ids") or r.get("gen_ids") is None or r.get("finish_reason") is None for r in rows):
        why.append("record without prompt_ids / gen_ids / finish_reason")
    return why


def plan_cells(exp: dict, role: str, recs: dict[str, dict[str, dict]], render) -> tuple[list[dict], list[str]]:
    """Cells for one role. recs[cond][id] = main-run record. render(messages) -> prompt ids.
    Returns (jobs, problems). Each job reads `gen_ids` of text_cond under read_cond's prompt."""
    qs = list(exp["pilot"]["questions"])
    reqs = {(r.question_id, r.condition): r for r in build_requests(exp, "main", [role], include_default=False)
            if r.prompt_variant == 0 and r.question_id in qs}
    jobs, problems = [], []
    rendered = {}
    for (qid, cond), rq in reqs.items():
        rec = recs[cond].get(rq.id)
        if rec is None:
            problems.append(f"missing {rq.id}")
            continue
        ids = render(list(rq.messages))
        if ids != list(rec["prompt_ids"]):
            problems.append(f"re-rendered prompt != stored prompt_ids for {rq.id}")
            continue
        rendered[(qid, cond)] = ids
    groups = {q["id"]: q["group"] for q in _questions(exp)}
    for qid in qs:
        g = groups[qid]
        graded = [c for c in exp["condition_sets"][g] if c != "baseline"]
        if (qid, "baseline") not in rendered:
            continue
        base_rq = reqs[(qid, "baseline")]
        base_rec = recs["baseline"][base_rq.id]
        add = lambda text_c, text_rec, read_c, c: jobs.append({  # noqa: E731
            "cell_id": f"{role}|{qid}|{c}|text={text_c}|read={read_c}", "role": role, "question_id": qid,
            "question_group": g, "condition": c, "text": "graded" if text_c != "baseline" else "baseline",
            "read": "graded" if read_c != "baseline" else "baseline", "wording_id": reqs[(qid, c)].wording_id,
            "prompt_ids": rendered[(qid, read_c)], "gen_ids": list(text_rec["gen_ids"]),
            "source_id": text_rec["id"], "has_answer": bool(text_rec.get("has_answer"))})
        add("baseline", base_rec, "baseline", "baseline")
        for c in graded:
            if (qid, c) not in rendered:
                continue
            grec = recs[c][reqs[(qid, c)].id]
            add("baseline", base_rec, c, c)       # baseline text, graded context
            add(c, grec, "baseline", c)           # graded text, baseline context
            add(c, grec, c, c)                    # graded text, graded context (own)
    return jobs, problems


def _questions(exp: dict) -> list[dict]:
    from ..config import load_questions

    return load_questions(exp)


def decompose(h: dict[tuple, np.ndarray]) -> dict[str, np.ndarray]:
    """h[(text, read)] -> context / content effects for one unit and condition (A.4)."""
    bb, bg = h[("baseline", "baseline")], h[("baseline", "graded")]
    gb, gg = h[("graded", "baseline")], h[("graded", "graded")]
    return {"CE_B": bg - bb, "CE_G": gg - gb, "TE_base": gb - bb, "TE_c": gg - bg, "Delta": gg - bb,
            "I": (gg - gb) - (bg - bb)}
