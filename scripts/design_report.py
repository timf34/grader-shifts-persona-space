#!/usr/bin/env python3
"""Print (and optionally save) the frozen design: request counts per stage and model,
condition / wording balance, splits, and the config id that guards each output root.
CPU only, no model or tokenizer needed. Run it before launching anything on a pod.

    python scripts/design_report.py
    python scripts/design_report.py --json design_report.json
"""

import argparse
import collections
import copy
import json

from _common import load_experiment

from gshift.config import heldout_roles, load_models, load_questions, manifest_sha, role_categories
from gshift.design import build_requests
from gshift.generate import config_identity


def stage_summary(exp: dict, stage: str, enable_thinking: bool | None = None) -> dict:
    reqs = build_requests(exp, stage, enable_thinking=enable_thinking)
    c = collections.Counter
    graded = [r for r in reqs if r.condition != "baseline"]
    blocks = {(r.question_group, r.question_split, r.question_id, r.prompt_variant, r.wording_id)
              for r in graded if r.role != exp["default_role"]}
    balance = collections.defaultdict(collections.Counter)
    for g, split, _, _, w in blocks:
        balance[f"{g}|{split}"][w] += 1
    return {
        "n_requests": len(reqs),
        "n_prompted": sum(r.role != exp["default_role"] for r in reqs),
        "n_default_control": sum(r.role == exp["default_role"] for r in reqs),
        "by_group_condition": dict(c(f"{r.question_group}|{r.condition}" for r in reqs)),
        "by_wording": dict(c(r.wording_id for r in reqs)),
        "by_role_split": dict(c(r.role_split for r in reqs)),
        "by_question_split": dict(c(r.question_split for r in reqs)),
        "roles": len({r.role for r in reqs}), "prompt_variants": sorted({r.prompt_variant for r in reqs}),
        "questions": len({r.question_id for r in reqs}),
        "wording_balance_per_block": {k: dict(v) for k, v in sorted(balance.items())},
        "baselines_reused_from_main": len({r.baseline_id for r in reqs if r.baseline_id}
                                          - {r.id for r in reqs}),
        "enable_thinking": sorted({r.enable_thinking for r in reqs}),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", default=None)
    p.add_argument("--json", default=None, help="also write the report here")
    args = p.parse_args()
    exp = load_experiment(args.experiment)
    models = load_models()
    cats = role_categories(exp)
    held = heldout_roles(exp)
    q = load_questions(exp)
    rep = {
        "design_version": exp["design_version"], "frozen_on": exp["frozen_on"],
        "question_manifest": exp["question_manifest"], "question_manifest_sha": manifest_sha(exp),
        "roles": {"n": len(cats), "discovery": [r for r in cats if r not in held], "heldout": held,
                  "by_category": {k: list(v) for k, v in exp["roles"].items()}},
        "questions": dict(collections.Counter(f"{x['group']}|{x['split']}" for x in q)),
        "stages": {st: stage_summary(exp, st) for st in exp["stages"]},
        "pilot": {f"thinking_{'on' if t else 'off'}": stage_summary(exp, "pilot", t)
                  for t in exp["pilot"]["thinking_arms"]},
        "config_ids": {},
    }
    for key, spec in models.items():
        cid, _ = config_identity(spec, exp)
        ids = {"main_grid": cid}
        for t in exp["pilot"]["thinking_arms"]:
            e = copy.deepcopy(exp)
            e["generation"]["enable_thinking"] = bool(t)
            e["generation"]["max_tokens"] = int(exp["pilot"]["max_tokens"][t])
            ids[f"pilot_thinking_{'on' if t else 'off'}"] = config_identity(spec, e)[0]
        rep["config_ids"][key] = ids
    n_models = len(models)
    tot = {st: s["n_requests"] for st, s in rep["stages"].items()}
    pil = sum(s["n_requests"] for s in rep["pilot"].values())

    print(f"design v{rep['design_version']} (frozen {rep['frozen_on']}), manifest {rep['question_manifest_sha']}")
    print(f"roles: {len(cats)} ({len(cats) - len(held)} discovery / {len(held)} held out: {', '.join(held)})")
    print(f"questions: {rep['questions']}")
    print(f"\n{'stage':22s} {'per model':>10s} {'x' + str(n_models) + ' models':>12s}")
    for st, n in tot.items():
        print(f"{st:22s} {n:>10,d} {n * n_models:>12,d}")
    for arm, s in rep["pilot"].items():
        print(f"{'pilot ' + arm:22s} {s['n_requests']:>10,d} {s['n_requests'] * n_models:>12,d}")
    print(f"{'pilot total':22s} {'':>10s} {pil * n_models:>12,d}")
    for st, s in rep["stages"].items():
        print(f"\n[{st}] conditions: {s['by_group_condition']}")
        print(f"[{st}] wordings: {s['by_wording']}")
        if s["wording_balance_per_block"]:
            print(f"[{st}] wording balance per question x prompt block: {s['wording_balance_per_block']}")
        if s["baselines_reused_from_main"]:
            print(f"[{st}] reuses {s['baselines_reused_from_main']:,d} main-grid baseline records (not regenerated)")
    print("\nconfig ids (guard resumption; differ => separate output root):")
    for key, ids in rep["config_ids"].items():
        print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in ids.items()))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
