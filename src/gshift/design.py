"""The experimental grid: which conversations get generated, and their metadata.

A *unit* is (role, prompt variant, question id, sample). The same unit is generated under
every condition, with the same sampling seed (common random numbers), so paired rows
differ only in the one context sentence. Request ids carry no stage, so the baseline
generated once in `main` is the matched baseline for every stage.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from .config import condition_texts, load_questions, load_role, role_categories


@dataclass(frozen=True)
class Request:
    id: str
    stage: str
    role: str
    category: str                  # role category ("default" for the control)
    role_split: str                # discovery | heldout | control
    prompt_variant: int
    question_id: str               # stable manifest id
    question_group: str            # original | checkable
    question_domain: str
    question_family: str
    question_split: str            # discovery | validation
    condition: str
    sample: int
    seed: int
    system_prompt: str | None      # None = no system message
    question: str                  # ORIGINAL question (used for judging/matching)
    prefix: str | None             # context statement (None for baseline)
    user_content: str              # actual user message (used for generation/replay)
    messages: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["messages"] = [dict(m) for m in self.messages]
        return d


def unit_seed(base_seed: int, role: str, prompt_variant: int, question_id: str, sample: int) -> int:
    """Stable per-unit seed, shared by all conditions (and stages) of a unit."""
    key = f"{base_seed}|{role}|{prompt_variant}|{question_id}|{sample}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") & 0x7FFFFFFF


def request_id(role: str, pv: int, qid: str, cond: str, sample: int) -> str:
    return f"{role}|p{pv}|{qid}|{cond}|s{sample}"


def build_messages(system_prompt: str | None, question: str, statement: str | None,
                   placement: str) -> tuple[str | None, str]:
    """Return (system, user). Plan default: user = "<prefix>\\n\\n<question>"."""
    system = system_prompt if system_prompt else None
    user = question
    if statement:
        if placement == "user_prefix":
            user = f"{statement}\n\n{question}"
        elif placement == "user_suffix":
            user = f"{question}\n\n{statement}"
        elif placement == "system_suffix":
            system = f"{system} {statement}" if system else statement
        else:
            raise ValueError(f"unknown placement {placement!r}")
    return system, user


def role_instruction(role: str, variant: int, model_short_name: str = "Qwen") -> str:
    text = load_role(role)["instruction"][variant]["pos"]
    return text.replace("{model_name}", model_short_name)   # literal, never .format


def _select_questions(rows: list[dict], which: str) -> list[dict]:
    if which == "all":
        return rows
    if which in ("discovery", "validation"):
        return [r for r in rows if r["split"] == which]
    raise ValueError(f"unknown question selection {which!r}")


def iter_requests(exp: dict[str, Any], stage: str, roles: list[str] | None = None,
                  include_default: bool = True) -> Iterator[Request]:
    st = exp["stages"][stage]
    texts = condition_texts(exp)
    for c in st["conditions"]:
        if c not in texts:
            raise KeyError(f"stage {stage}: condition {c!r} has no text defined")
    cats = role_categories(exp)
    heldout = set(exp["role_split"]["heldout"])
    questions = _select_questions(load_questions(exp), st.get("questions", "all"))
    placement = exp.get("placement", "user_prefix")
    gen = exp["generation"]
    n_samples = int(gen.get("samples_per_prompt", 1))

    role_list: list[tuple[str, str, str, list[int]]] = [
        (r, cats[r], "heldout" if r in heldout else "discovery", list(st["prompt_variants"])) for r in cats
    ]
    if include_default:
        dv = [v for v in exp["default_prompt_variants"] if v in st["prompt_variants"]] \
            or list(exp["default_prompt_variants"])[:1]
        role_list.append((exp["default_role"], "default", "control", dv))
    if roles is not None:
        role_list = [x for x in role_list if x[0] in roles]

    for role, cat, rsplit, variants in role_list:
        for pv in variants:
            system_prompt = role_instruction(role, pv)
            for q in questions:
                for s in range(n_samples):
                    seed = unit_seed(int(gen["base_seed"]), role, pv, q["id"], s)
                    for cond in st["conditions"]:
                        statement = texts[cond]
                        system, user = build_messages(system_prompt, q["text"], statement, placement)
                        msgs = ([{"role": "system", "content": system}] if system else []) + [
                            {"role": "user", "content": user}]
                        yield Request(
                            id=request_id(role, pv, q["id"], cond, s), stage=stage, role=role, category=cat,
                            role_split=rsplit, prompt_variant=pv, question_id=q["id"],
                            question_group=q["group"], question_domain=q["domain"], question_family=q["family"],
                            question_split=q["split"], condition=cond, sample=s, seed=seed,
                            system_prompt=system_prompt or None, question=q["text"], prefix=statement,
                            user_content=user, messages=tuple(msgs),
                        )


def build_requests(exp: dict[str, Any], stage: str, roles: list[str] | None = None,
                   include_default: bool = True) -> list[Request]:
    reqs = list(iter_requests(exp, stage, roles, include_default))
    ids = [r.id for r in reqs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate request ids in design")
    return reqs


def partition_roles(roles: list[str], worker: int, n_workers: int) -> list[str]:
    """Deterministic role partition for data-parallel workers (round robin over sorted roles)."""
    return [r for i, r in enumerate(sorted(roles)) if i % n_workers == worker]
