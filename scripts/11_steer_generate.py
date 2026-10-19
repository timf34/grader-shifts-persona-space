#!/usr/bin/env python3
"""Q2 step 2 (client of a running EasySteer server): sanity check, strength calibration,
steered evaluation.

    bash scripts/pod/serve_easysteer.sh qwen3.6-27b          # in the EasySteer venv
    python -u scripts/11_steer_generate.py --model qwen3.6-27b --phase all
    python -u scripts/11_steer_generate.py --model qwen3.6-27b --phase eval --strengths 0.1 0.2

Phases:
  sanity     greedy: scale 0 == unsteered and a large scale changes the output (else stop)
  calibrate  calibration prompts under none + primary direction at +-grid -> calibration.json
             with s* chosen by the pre-registered rule (configs/steering.yaml)
  eval       benchmarks x conditions -> steering/gen/<benchmark>/<condition>.jsonl
Resumable per (benchmark, condition, prompt).
"""

import json
import sys

from _common import ROOT, base_parser, experiment_for

from gshift.config import get_model, load_yaml
from gshift.generate import load_tokenizer
from gshift.io import write_json
from gshift.steer.calibrate import calib_metrics, choose_strength, load_calibration_rows
from gshift.steer.engine import SteerClient, condition_name, generate_condition, sanity_check
from gshift.steer.tasks import benchmark_prompts, calibration_prompts, natural_prefix
from gshift.steer.vectors import gguf_paths, load_steering_set, steering_dir


def eval_conditions(cfg: dict, man: dict, strengths: list[float]) -> list[str]:
    """none + natural_* + recovered directions at +-{s*/2, s*} + random and decomposition
    (inside / outside the baseline PCA subspace) directions at +-s*."""
    ev = cfg["evaluation"]
    conds = list(ev["conditions"])
    for d in ev["steered_directions"]:
        if d in man["directions"]:
            conds += [condition_name(d, sg * s) for s in strengths for sg in (1, -1)]
    s_max = max(strengths)
    for d, info in man["directions"].items():
        if info["kind"] == "random":
            conds += [condition_name(d, sg * s_max) for sg in ev.get("random_signs", [1])]
        if info["kind"] == "projection":
            conds += [condition_name(d, sg * s_max) for sg in ev.get("projection_signs", [1, -1])]
    return conds


def main():
    p = base_parser(__doc__)
    p.add_argument("--steering-config", default=str(ROOT / "configs" / "steering.yaml"))
    p.add_argument("--phase", choices=["sanity", "calibrate", "eval", "all"], default="all")
    p.add_argument("--strengths", nargs="+", type=float, default=None, help="override calibrated strengths")
    p.add_argument("--benchmarks", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None, help="prompts per benchmark (smoke)")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--concurrency", type=int, default=128)
    args = p.parse_args()
    exp = experiment_for(args)
    cfg = load_yaml(args.steering_config)
    if args.smoke:
        args.limit = args.limit or 4
        cfg["calibration"]["strength_grid"] = cfg["calibration"]["strength_grid"][:2]
        cfg["generation"]["max_tokens"] = 1024
    spec = get_model(args.model)
    man, _ = load_steering_set(spec.key)
    paths = gguf_paths(spec.key, man)
    sd = steering_dir(spec.key)
    tok = load_tokenizer(spec)
    client = SteerClient(args.base_url, spec.key, args.concurrency)
    gen = cfg["generation"]
    primary = cfg["calibration"]["primary_direction"]

    if args.phase in ("sanity", "all"):
        res = sanity_check(client, tok, spec, paths[primary], int(man["layer"]), float(man["activation_norm"]))
        write_json(sd / "sanity.json", res)
        print("[sanity]", json.dumps({k: v for k, v in res.items() if not k.endswith("_text")}), flush=True)
        if not res["ok"]:
            sys.exit("EasySteer sanity check failed: steering is not being applied as expected")

    if args.phase in ("calibrate", "all"):
        prompts = calibration_prompts(cfg, exp)
        prompts = prompts[: args.limit] + prompts[-args.limit:] if args.limit else prompts
        out = sd / "calibration_generations.jsonl"
        conds = ["none"] + [condition_name(primary, sg * s) for s in cfg["calibration"]["strength_grid"]
                            for sg in (1, -1)]
        for c in conds:
            generate_condition(client, tok, spec, prompts, c, paths, man, gen, out)
        rows = load_calibration_rows(out)
        table = {c: calib_metrics(rows.get(c, [])) for c in conds}
        choice = choose_strength(table, cfg, primary)
        write_json(sd / "calibration.json", {"table": table, **choice})
        print("[calibrate]", json.dumps(choice), flush=True)

    if args.phase in ("eval", "all"):
        strengths = args.strengths or json.loads((sd / "calibration.json").read_text())["eval_strengths"]
        conds = eval_conditions(cfg, man, strengths)
        write_json(sd / "eval_plan.json", {"strengths": strengths, "conditions": conds})
        ac = cfg["activation_check"]
        s_star = max(strengths)
        persona_dirs = list(ac["directions"]) + [f"random{k}" for k in range(int(ac.get("n_random", 0)))]
        persona_conds = {"none"} | {condition_name(d, sg * s_star) for d in persona_dirs for sg in (1, -1)}
        for b in args.benchmarks or list(cfg["evaluation"]["benchmarks"]):
            for c in sorted(persona_conds & (set(conds) | {"none"})) if b == "persona" else conds:
                if b == "persona" and c.startswith("natural_"):
                    continue
                prompts = benchmark_prompts(b, cfg, exp, natural_prefix(c, exp))
                if args.limit:
                    prompts = prompts[: args.limit]
                generate_condition(client, tok, spec, prompts, c, paths, man, gen, sd / "gen" / b / f"{c}.jsonl")


if __name__ == "__main__":
    main()
