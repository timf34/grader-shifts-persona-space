"""Prompt sets for Q2: calibration, School of Reward Hacks, GSM8K, the held-out behavioural
suites (TruthfulQA MC1, open-ended traits: extra_benchmarks.py) and the persona check.
Calibration prompts are checked to share no item with any held-out suite."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .. import REPO_ROOT
from ..config import condition_texts, load_questions
from ..design import build_requests

BANNED = ("grade", "graded", "score", "evaluat", "cheat", "reward", "test", "exam", "judge", "assess", "review")


@dataclass
class Prompt:
    id: str
    benchmark: str
    messages: list[dict[str, str]]
    meta: dict[str, Any] = field(default_factory=dict)


def _jsonl(path: str) -> list[dict]:
    p = REPO_ROOT / path if not str(path).startswith("/") else path
    return [json.loads(l) for l in open(p) if l.strip()]


def with_prefix(text: str, prefix: str | None) -> str:
    return f"{prefix}\n\n{text}" if prefix else text


def calibration_prompts(cfg: dict, exp: dict) -> list[Prompt]:
    """Ordinary Assistant Axis questions NOT in the Q1 manifest + GSM8K train problems."""
    used = {int(q["id"][7:]) for q in load_questions(exp) if q["group"] == "original"}
    aa = [json.loads(l) for l in open(REPO_ROOT / "data" / "aa_extraction_questions_all.jsonl")]
    pool = [r for r in aa if int(r["id"]) not in used and not any(b in r["question"].lower() for b in BANNED)]
    rng = np.random.default_rng(int(cfg["calibration"]["aa_question_seed"]))
    pick = [pool[i] for i in sorted(rng.choice(len(pool), int(cfg["calibration"]["n_aa_questions"]), replace=False))]
    out = [Prompt(f"calib_aa{r['id']:03d}", "calib_open", [{"role": "user", "content": r["question"]}]) for r in pick]
    instr = cfg["evaluation"]["gsm8k_instruction"]
    for r in _jsonl(cfg["calibration"]["gsm8k_file"]):
        out.append(Prompt(r["id"], "calib_gsm8k", [{"role": "user", "content": f"{instr}\n\n{r['question']}"}],
                          {"answer": r["answer"]}))
    assert_disjoint_from_heldout(out, cfg)
    return out


def heldout_texts(cfg: dict) -> dict[str, set[str]]:
    """Item texts of the held-out behavioural suites (never used for calibration)."""
    from .extra_benchmarks import load_trait_questions, load_truthfulqa

    out = {"truthfulqa": {r["question"].strip() for r in load_truthfulqa()},
           "trait_openended": {q["question"].strip() for q in load_trait_questions()}}
    ev = cfg["evaluation"]["benchmarks"]
    if "gsm8k" in ev:
        out["gsm8k_test"] = {r["question"].strip() for r in _jsonl(ev["gsm8k"]["file"])}
    if "sorh" in ev:
        out["sorh"] = {r["user_descored"].strip() for r in _jsonl(ev["sorh"]["file"])}
    return out


def assert_disjoint_from_heldout(prompts: list[Prompt], cfg: dict) -> None:
    held = heldout_texts(cfg)
    for p in prompts:
        text = p.messages[-1]["content"]
        for name, items in held.items():
            hit = next((q for q in items if q and q in text), None)
            if hit is not None:
                raise ValueError(f"calibration prompt {p.id} contains a held-out {name} item: {hit[:80]!r}")


def benchmark_prompts(name: str, cfg: dict, exp: dict, prefix: str | None = None) -> list[Prompt]:
    b = cfg["evaluation"]["benchmarks"].get(name) or {}
    if name in ("sorh", "sorh_descored"):
        rows = _jsonl(b["file"])
        return [Prompt(f"srh{r['id']}", name, [{"role": "user", "content": with_prefix(r[b["field"]], prefix)}],
                       {"user": r["user"], "user_descored": r["user_descored"], "cat": r["cat"]}) for r in rows]
    if name == "gsm8k":
        instr = cfg["evaluation"]["gsm8k_instruction"]
        return [Prompt(r["id"], name, [{"role": "user", "content": with_prefix(f"{instr}\n\n{r['question']}", prefix)}],
                       {"answer": r["answer"]}) for r in _jsonl(b["file"])]
    if name == "truthfulqa":
        from .extra_benchmarks import truthfulqa_prompts

        return truthfulqa_prompts(cfg, prefix)
    if name == "trait_openended":
        from .extra_benchmarks import trait_prompts

        return trait_prompts(cfg, prefix)
    if name.startswith("choice_"):
        from .persona_choice import choice_prompts

        return choice_prompts(name, cfg, exp, prefix)
    if name.startswith("proxy_") and not b.get("depends_on"):
        from .proxy_panel import proxy_prompts

        return proxy_prompts(name, cfg, exp, prefix)
    if name in ("adherence", "adherence_graded"):
        return adherence_prompts(name, b, exp, prefix)
    if name == "agentic":
        from .agentic import agentic_prompts

        return agentic_prompts(cfg, prefix)
    if name == "persona":
        # held-out roles x validation questions, prompt variant 0, baseline (no grading prefix)
        splits = {"heldout"} if b.get("roles", "heldout") == "heldout" else {"heldout", "discovery"}
        reqs = [r for r in build_requests(exp, "main", include_default=False)
                if r.role_split in splits and r.question_split == "validation"
                and r.prompt_variant == 0 and r.condition == "baseline"]
        return [Prompt(r.id, name, [dict(m) for m in r.messages],
                       {"role": r.role, "question_id": r.question_id, "question": r.question}) for r in reqs]
    raise KeyError(name)


def adherence_prompts(name: str, b: dict, exp: dict, prefix: str | None = None) -> list[Prompt]:
    """Held-out roles x validation questions (Q1 main grid, baseline requests: the role's
    system prompt and the question). `adherence` sends them as they are; `adherence_graded`
    adds the group's graded statement (b["graded_condition"], wording b["wording"]) as the
    Q1 user prefix, so steering against the grading direction can be tested under real
    grading. Ids are shared between the two benchmarks (same sampling seeds, paired)."""
    from ..config import statement

    pvs = {g: set(v) for g, v in b.get("group_prompt_variants", {"checkable": [0], "original": [0]}).items()}
    reqs = [r for r in build_requests(exp, "main", include_default=False)
            if r.role_split == "heldout" and r.question_split == "validation" and r.condition == "baseline"
            and r.prompt_variant in pvs.get(r.question_group, set())]
    out = []
    for r in reqs:
        msgs = [dict(m) for m in r.messages]
        graded = None
        if name == "adherence_graded":
            graded = b["graded_condition"][r.question_group]
            msgs[-1]["content"] = with_prefix(msgs[-1]["content"], statement(exp, graded, b.get("wording", "w1")))
        if prefix:
            msgs[-1]["content"] = with_prefix(msgs[-1]["content"], prefix)
        out.append(Prompt(f"adh_{r.role}_p{r.prompt_variant}_{r.question_id}", name, msgs,
                          {"role": r.role, "question_id": r.question_id, "question": r.question,
                           "group": r.question_group, "prompt_variant": r.prompt_variant, "graded_condition": graded}))
    return out


def natural_prefix(condition: str, exp: dict) -> str | None:
    if condition.startswith("natural_"):
        return condition_texts(exp)[condition[len("natural_"):]]
    return None


# ---------------------------------------------------------------------------
# GSM8K scoring
# ---------------------------------------------------------------------------

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_number(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    cand = boxed[-1] if boxed else text or ""
    nums = _NUM.findall(cand.replace("$", ""))
    if not nums:
        return None
    return nums[-1].replace(",", "").rstrip(".")


def gsm8k_correct(answer_text: str, gold: str) -> bool | None:
    got = extract_number(answer_text)
    if got is None:
        return None
    try:
        return abs(float(got) - float(gold)) < 1e-6
    except ValueError:
        return None
