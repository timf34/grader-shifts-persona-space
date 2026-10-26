"""Prompt sets and generation jobs per stage (pilot / confirm), shared by the pod pipeline and
the local analysis so that ids, contexts and outcomes are defined in one place.

Benchmarks (output gen/<benchmark>/<condition>.jsonl):
  rpt_<ctx>        Track A: every persona submits the report (assay.build "rpt")
  rptq_<ctx>       Track A: questionnaire version, conflict items only
  choice_constrained_<ctx>   Track A (confirm): existing forced-choice items, selected dimensions
  steer_<ctx>      Track B: the default Assistant submits the report under steering conditions
Conditions: "none" or "<dir>@L<layer>_<+|-><strength>" with dir in {trait, random<k>, traitdpo}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .. import REPO_ROOT
from ..config import load_yaml
from ..design import role_instruction
from ..steer.tasks import Prompt
from . import assay

CFG_FILE = REPO_ROOT / "configs" / "psm.yaml"


def load_cfg() -> dict[str, Any]:
    return load_yaml(CFG_FILE)


def personas(cfg: dict) -> list[str]:
    return json.loads((REPO_ROOT / cfg["personas_file"]).read_text())["personas"]


def system_for(persona: str) -> str:
    return "" if persona == "default" else role_instruction(persona, 0)


def family_of(model_key: str) -> str:
    return "olmo" if model_key.startswith("olmo") else "qwen"


def gen_cfg(cfg: dict, model_key: str) -> dict[str, Any]:
    return dict(cfg["generation"][family_of(model_key)])


def cond_name(direction: str, layer: int, signed: float) -> str:
    return f"{direction}@L{layer}_{'+' if signed >= 0 else '-'}{abs(signed):g}"


def parse_cond(name: str) -> tuple[str | None, int | None, float]:
    if name == "none":
        return None, None, 0.0
    head, s = name.rsplit("_", 1)
    d, L = head.split("@L")
    return d, int(L), float(s)


@dataclass
class Job:
    bench: str
    condition: str
    prompts: list[Prompt]


def _pressures(cfg: dict) -> set[str] | None:
    fr = cfg.get("frozen") or {}
    return set(fr["pressures"]) if fr.get("pressures") else None


def _filter(items: list[dict], keep_pressures: set[str] | None) -> list[dict]:
    if keep_pressures is None:
        return items
    return [it for it in items if it["kind"] != "conflict" or it["pressure"] in keep_pressures]


def track_a_jobs(cfg: dict, stage: str) -> list[Job]:
    st = cfg[stage]["track_a"]
    split = "discovery" if stage == "pilot" else "confirm"
    items = _filter(assay.items_for(split), _pressures(cfg) if stage == "confirm" else None)
    ps = personas(cfg)
    jobs = []
    for ctx in st["contexts"]:
        jobs.append(Job(f"rpt_{ctx}", "none", assay.build("rpt", items, ps, [ctx], int(st["samples"]), system_for)))
    for ctx in st.get("questionnaire_contexts", st["contexts"]):
        jobs.append(Job(f"rptq_{ctx}", "none",
                        assay.build("rptq", items, ps, [ctx], int(st["questionnaire_samples"]), system_for)))
    fc = st.get("forced_choice")
    if fc:
        from ..steer.persona_choice import choice_prompts

        for ctx in fc["contexts"]:
            bench = f"choice_{fc['format']}_{ctx}"
            pcfg = {"evaluation": {"benchmarks": {bench: {"samples": int(fc["samples"]), "personas": ps}}}}
            pr = [p for p in choice_prompts(bench, pcfg, {"default_role": "default"})
                  if p.meta["dimension"] in set(fc["dimensions"])]
            jobs.append(Job(bench, "none", pr))
    return jobs


def steer_prompts(cfg: dict, stage: str, ctx: str, samples: int) -> list[Prompt]:
    split = "discovery" if stage == "pilot" else "confirm"
    items = _filter(assay.items_for(split), _pressures(cfg) if stage == "confirm" else None)
    pr = assay.build("rpt", items, ["default"], [ctx], samples, system_for)
    for p in pr:
        p.benchmark = f"steer_{ctx}"
    return pr


def sweep_jobs(cfg: dict) -> list[Job]:
    sw = cfg["steering"]["sweep"]
    pr = steer_prompts(cfg, "pilot", "base", int(sw["samples"]))
    conds = ["none"]
    for L in sw["layers"]:
        for s in sw["strengths"]:
            conds += [cond_name("trait", L, sg * s) for sg in (1, -1)]
        for k in range(int(sw["random"]["n"])):
            for s in sw["random"]["strengths"]:
                conds += [cond_name(f"random{k}", L, sg * s) for sg in (1, -1)]
    return [Job("steer_base", c, pr) for c in conds]


def confirm_steer_jobs(cfg: dict, choice: dict[str, Any], transfer: bool = False) -> list[Job]:
    """choice = {"layer": L, "strength": s}: trait at +-s/2, +-s; random at +-s; contexts x."""
    tb = cfg["confirm"]["track_b"]
    L, s = int(choice["layer"]), float(choice["strength"])
    conds = ["none"] + [cond_name("trait", L, sg * x) for x in (s / 2, s) for sg in (1, -1)]
    conds += [cond_name(f"random{k}", L, sg * s) for k in range(int(tb["random"]["n"])) for sg in (1, -1)]
    if transfer:
        conds += [cond_name("traitdpo", L, sg * s) for sg in (1, -1)]
    jobs = []
    for ctx in tb["contexts"]:
        pr = steer_prompts(cfg, "confirm", ctx, int(tb["samples"]))
        jobs += [Job(f"steer_{ctx}", c, pr) for c in conds]
    return jobs


def stage_jobs(cfg: dict, stage: str, choice: dict | None = None, transfer: bool = False) -> list[Job]:
    if stage == "pilot":
        return track_a_jobs(cfg, "pilot") + sweep_jobs(cfg)
    if stage == "confirm":
        if choice is None:
            raise ValueError("confirm needs the frozen steering choice")
        return track_a_jobs(cfg, "confirm") + confirm_steer_jobs(cfg, choice, transfer)
    raise ValueError(stage)


def activation_prompts(cfg: dict, stage: str) -> list[Prompt]:
    """Unique (persona, item, context) report prompts of a stage, sample 0 only (the prompt does
    not depend on the sample): the persona landscape ON the assay prompts, and the trait
    projection before the decision."""
    seen, out = set(), []
    for j in track_a_jobs(cfg, stage):
        if not j.bench.startswith("rpt_"):
            continue
        for p in j.prompts:
            if p.meta["sample"] != 0:
                continue
            key = (p.benchmark, p.id)
            if key not in seen:
                seen.add(key)
                out.append(p)
    return out
