"""The experimental grid: which conversations get generated, and their metadata.

A *block* is (question, prompt variant); a *unit* is (role, prompt variant, question,
sample). Every condition of a unit uses the same sampling seed (common random numbers).
Original questions get [baseline, auto, human]; checkable questions get [baseline,
auto_correct, human_correct, auto_quality, human_quality] (configs/experiment.yaml
condition_sets). In the main grid each block carries ONE discovery wording template
(deterministic, balanced; identical across roles, conditions and models), and baseline is
generated once per block. Every graded request names its matched baseline record
(`baseline_id`, always the main-grid record of the same unit): the wording-validation and
follow-up stages reuse it rather than regenerating it. Request ids include the wording, so
the reserved validation wordings can never collide with (or be mistaken for) discovery
outputs.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterator

from .config import condition_def, heldout_roles, load_questions, load_role, role_categories, statement


@dataclass(frozen=True)
class Request:
    id: str
    stage: str
    role: str
    category: str                  # role category ("default" for the control)
    role_split: str                # discovery | heldout | control
    prompt_variant: int
    question_id: str
    question_group: str            # original | checkable
    question_domain: str
    question_family: str
    question_split: str            # discovery | validation
    condition: str
    evaluator: str | None          # automated | human | None
    criterion: str | None          # unspecified | correctness | quality | evaluation_awareness | None
    wording_id: str                # w1..w3 (discovery), v1..v2 (validation), "none" (baseline), "eval"
    wording_set: str               # discovery | validation | none | followup
    baseline_id: str | None        # id of the matched MAIN-grid baseline record (None for baselines);
                                   # reused, never regenerated, by wording_validation / followup
    enable_thinking: bool
    sample: int
    seed: int
    system_prompt: str | None      # None = no system message
    question: str                  # ORIGINAL question (judging/matching)
    prefix: str | None             # context statement (None for baseline)
    user_content: str              # actual user message (generation/replay)
    messages: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["messages"] = [dict(m) for m in self.messages]
        return d


def unit_seed(base_seed: int, role: str, prompt_variant: int, question_id: str, sample: int) -> int:
    """Stable per-unit seed, shared by all conditions (and stages) of a unit."""
    key = f"{base_seed}|{role}|{prompt_variant}|{question_id}|{sample}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") & 0x7FFFFFFF


def request_id(role: str, pv: int, qid: str, cond: str, wording: str, sample: int) -> str:
    return f"{role}|p{pv}|{qid}|{cond}|{wording}|s{sample}"


def build_messages(system_prompt: str | None, question: str, statement_text: str | None,
                   placement: str) -> tuple[str | None, str]:
    """Return (system, user). Plan default: user = "<statement>\\n\\n<question>"."""
    system = system_prompt if system_prompt else None
    user = question
    if statement_text:
        if placement == "user_prefix":
            user = f"{statement_text}\n\n{question}"
        elif placement == "user_suffix":
            user = f"{question}\n\n{statement_text}"
        elif placement == "system_suffix":
            system = f"{system} {statement_text}" if system else statement_text
        else:
            raise ValueError(f"unknown placement {placement!r}")
    return system, user


def role_instruction(role: str, variant: int, model_short_name: str = "Qwen") -> str:
    text = load_role(role)["instruction"][variant]["pos"]
    return text.replace("{model_name}", model_short_name)   # literal, never .format


def question_ranks(questions: list[dict], keys: tuple[str, ...] = ("group", "split")) -> dict[str, int]:
    """Rank of each question within its (group, split) (or other `keys`), in manifest order."""
    counters: dict[tuple, int] = {}
    out = {}
    for q in questions:
        k = tuple(q[x] for x in keys)
        out[q["id"]] = counters.get(k, 0)
        counters[k] = out[q["id"]] + 1
    return out


def discovery_wording(exp: dict[str, Any], rank: int, prompt_variant: int) -> str:
    """Balanced deterministic assignment: discovery[(rank + prompt_variant) mod n]."""
    ids = list(exp["wording_templates"]["discovery"])
    if exp.get("wording_assignment", "rank_plus_variant_mod_3") != "rank_plus_variant_mod_3":
        raise ValueError(f"unknown wording_assignment {exp.get('wording_assignment')!r}")
    return ids[(rank + prompt_variant) % len(ids)]


def default_wording(exp: dict[str, Any], group_rank: int) -> str:
    """Default-Assistant control (one prompt variant): its own question-level assignment,
    discovery[rank within question group mod n], balanced per group."""
    ids = list(exp["wording_templates"]["discovery"])
    if exp.get("default_wording_assignment", "group_rank_mod_3") != "group_rank_mod_3":
        raise ValueError(f"unknown default_wording_assignment {exp.get('default_wording_assignment')!r}")
    return ids[group_rank % len(ids)]


def _select_questions(rows: list[dict], which: str | list[str]) -> list[dict]:
    """'all' | 'discovery' | 'validation' | an explicit list of manifest ids (kept in
    manifest order)."""
    if isinstance(which, (list, tuple)):
        have = {r["id"] for r in rows}
        missing = [q for q in which if q not in have]
        if missing:
            raise ValueError(f"question ids not in the manifest: {missing}")
        keep = set(which)
        return [r for r in rows if r["id"] in keep]
    if which == "all":
        return rows
    if which in ("discovery", "validation"):
        return [r for r in rows if r["split"] == which]
    raise ValueError(f"unknown question selection {which!r}")


def _stage_conditions(exp: dict, st: dict, group: str) -> list[str]:
    if "conditions" in st:
        return list(st["conditions"])
    conds = list(exp["condition_sets"][group])
    if st.get("grading_only"):
        conds = [c for c in conds if c != "baseline"]
    return conds


def iter_requests(exp: dict[str, Any], stage: str, roles: list[str] | None = None,
                  include_default: bool = True, enable_thinking: bool | None = None) -> Iterator[Request]:
    """Requests of a stage. stage may be 'pilot' (uses the `pilot:` block)."""
    if stage == "pilot":
        pl = exp["pilot"]
        st = {"prompt_variants": pl["prompt_variants"], "questions": pl["questions"], "wording_set": "discovery"}
        roles = list(pl["roles"]) + ([exp["default_role"]] if include_default else []) if roles is None else roles
    else:
        st = exp["stages"][stage]
    thinking = bool(exp["generation"]["enable_thinking"]) if enable_thinking is None else bool(enable_thinking)
    cats = role_categories(exp)
    heldout = set(heldout_roles(exp))
    all_q = load_questions(exp)
    ranks = question_ranks(all_q)
    granks = question_ranks(all_q, ("group",))
    questions = _select_questions(all_q, st.get("questions", "all"))
    placement = exp.get("placement", "user_prefix")
    gen = exp["generation"]
    n_samples = int(gen.get("samples_per_prompt", 1))
    wset = st.get("wording_set")

    role_list: list[tuple[str, str, str, list[int]]] = [
        (r, cats[r], "heldout" if r in heldout else "discovery", list(st["prompt_variants"])) for r in cats]
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
                if wset == "discovery" and rsplit == "control":
                    wordings = [default_wording(exp, granks[q["id"]])]
                elif wset == "discovery":
                    wordings = [discovery_wording(exp, ranks[q["id"]], pv)]
                elif wset == "validation":
                    wordings = list(exp["wording_templates"]["validation"])
                else:
                    wordings = ["eval"]
                for s in range(n_samples):
                    seed = unit_seed(int(gen["base_seed"]), role, pv, q["id"], s)
                    for cond in _stage_conditions(exp, st, q["group"]):
                        for w in (["none"] if cond == "baseline" else wordings):
                            text = statement(exp, cond, None if cond == "baseline" or w == "eval" else w)
                            system, user = build_messages(system_prompt, q["text"], text, placement)
                            msgs = ([{"role": "system", "content": system}] if system else []) + [
                                {"role": "user", "content": user}]
                            d = condition_def(exp, cond)
                            yield Request(
                                id=request_id(role, pv, q["id"], cond, w, s), stage=stage, role=role, category=cat,
                                baseline_id=None if cond == "baseline" else request_id(role, pv, q["id"], "baseline",
                                                                                       "none", s),
                                role_split=rsplit, prompt_variant=pv, question_id=q["id"], question_group=q["group"],
                                question_domain=q["domain"], question_family=q["family"], question_split=q["split"],
                                condition=cond, evaluator=d["evaluator"], criterion=d["criterion"], wording_id=w,
                                wording_set="none" if w == "none" else ("followup" if w == "eval" else wset),
                                enable_thinking=thinking, sample=s, seed=seed, system_prompt=system_prompt or None,
                                question=q["text"], prefix=text, user_content=user, messages=tuple(msgs),
                            )


def build_requests(exp: dict[str, Any], stage: str, roles: list[str] | None = None,
                   include_default: bool = True, enable_thinking: bool | None = None) -> list[Request]:
    reqs = list(iter_requests(exp, stage, roles, include_default, enable_thinking))
    ids = [r.id for r in reqs]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate request ids in design")
    return reqs


def partition_roles(roles: list[str], worker: int, n_workers: int) -> list[str]:
    """Deterministic role partition for data-parallel workers (round robin over sorted roles)."""
    return [r for i, r in enumerate(sorted(roles)) if i % n_workers == worker]
