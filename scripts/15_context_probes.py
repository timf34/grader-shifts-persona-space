#!/usr/bin/env python3
"""Context probes (GPU, HF transformers, forward pass only): last-prompt-token residuals
for the persona probe (neutral / implicit-cue conditions vs Q1's grading statements) and
the Betley-thread control contrast pairs. Design: configs/probes.yaml (gshift.probes).

    python -u scripts/15_context_probes.py --model qwen3.6-27b
    python -u scripts/15_context_probes.py --model qwen3.6-27b --smoke          # 2 roles x 2 Q, 2 pairs x 4 tasks
    python -u scripts/15_context_probes.py --model qwen3.6-27b --only control

Writes outputs/<model>/probes/{persona,control}/ (resumable per condition / pair side) and
outputs/<model>/probes/summary.json. --smoke writes to outputs_smoke/ (never mixed with
real runs). Report on CPU afterwards: scripts/16_probe_report.py.
"""

import argparse
import json
import os
import time

from _common import ROOT

from gshift.config import get_model, load_experiment, require_pinned
from gshift.io import environment_info, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model key from configs/models.yaml")
    p.add_argument("--experiment", default=None, help="experiment.yaml (default configs/experiment.yaml)")
    p.add_argument("--probes-config", default=str(ROOT / "configs" / "probes.yaml"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--only", choices=["persona", "control"], default=None)
    p.add_argument("--layers", nargs="+", type=int, default=None, help="override capture_layers")
    p.add_argument("--max-batch-tokens", type=int, default=None)
    p.add_argument("--max-batch-size", type=int, default=None)
    p.add_argument("--no-self-check", action="store_true")
    args = p.parse_args()
    if args.smoke:
        os.environ["GSHIFT_OUTPUTS"] = os.environ.get("GSHIFT_SMOKE_OUTPUTS", str(ROOT / "outputs_smoke"))

    from gshift.extract import load_hf_model
    from gshift.probes import (control_prompts, load_probe_config, persona_prompts, probes_dir, quick_summary,
                               run_probe_set, smoke_subsets)
    from gshift.templates import inspect_template

    exp = load_experiment(args.experiment)
    pcfg = load_probe_config(args.probes_config)
    spec = get_model(args.model)
    if not args.smoke:
        require_pinned(spec)
    layers = sorted(set(args.layers or pcfg["capture_layers"]))
    run = pcfg.get("run", {})
    mbt = args.max_batch_tokens or int(run.get("max_batch_tokens", 32768))
    mbs = args.max_batch_size or int(run.get("max_batch_size", 64))
    thinking = bool(exp["generation"]["enable_thinking"])
    kw = spec.chat_template_kwargs

    sets = {}
    sub = smoke_subsets(exp, pcfg) if args.smoke else {}
    if args.only in (None, "persona"):
        sets["persona"] = persona_prompts(exp, pcfg, sub.get("roles"), sub.get("questions"))
    if args.only in (None, "control"):
        sets["control"] = control_prompts(pcfg, sub.get("pairs"), sub.get("tasks"))
    for s, files in sets.items():
        print(f"[probes] {s}: {len(files)} files x {len(next(iter(files.values())))} prompts", flush=True)

    t0 = time.time()
    model, tok = load_hf_model(spec, exp["extraction"].get("dtype", "bfloat16"))
    t_load = time.time() - t0
    info = inspect_template(tok, thinking, kw)
    if thinking and not info.prompt_opens_think:
        raise RuntimeError(f"generation prompt does not end inside <think>: {info.generation_prompt_suffix!r}")
    if info.notes:
        print(f"[probes] template notes: {info.notes}", flush=True)
    meta = {"model": spec.hf_id, "revision": spec.revision, "template": info.to_dict(),
            "probe_design_version": pcfg.get("probe_design_version"), "smoke": args.smoke,
            "environment": environment_info()}
    check = None if args.no_self_check else {"n": int(run.get("self_check_prompts", 4)),
                                             "min_cos": float(run.get("self_check_min_cos", 0.999))}
    summary = {"model": spec.key, "layers": layers, "smoke": args.smoke, "load_seconds": round(t_load, 1),
               "max_batch_tokens": mbt, "max_batch_size": mbs, "sets": {}}
    out_root = probes_dir(spec.key)
    for s, files in sets.items():
        t1 = time.time()
        st = run_probe_set(model, tok, files, out_root / s, layers, thinking, kw,
                           pcfg.get("save_dtype", "float16"), mbt, mbs, check, meta,
                           log=lambda m: print(m, flush=True))
        st["seconds"] = round(time.time() - t1, 1)
        st["n_prompts"] = sum(v["n"] for v in st["files"].values())
        st["n_tokens"] = sum(v["tokens"] for v in st["files"].values())
        ref = "baseline" if s == "persona" else None
        if ref:
            st["quick"] = quick_summary(out_root / s, list(files), ref, layers, int(pcfg.get("analysis_layer", 36)))
        summary["sets"][s] = st
        write_json(out_root / "summary.json", summary)
    summary["total_seconds"] = round(time.time() - t0, 1)
    write_json(out_root / "summary.json", summary)
    print(json.dumps({s: {k: v[k] for k in ("n_prompts", "n_tokens", "seconds", "self_check")}
                      for s, v in summary["sets"].items()}, indent=1), flush=True)


if __name__ == "__main__":
    main()
