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
from gshift.steer.engine import SteerClient, condition_name, generate_condition, parse_condition, sanity_check
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


def benchmark_conditions(cfg: dict, man: dict, strengths: list[float], bench: str) -> list[str]:
    """eval_conditions narrowed by the benchmark's optional filters (configs/steering_stage_a.yaml):
    directions, signs, strengths, natural (natural_* conditions), random."""
    b = cfg["evaluation"]["benchmarks"][bench] or {}
    keep_dirs = set(b.get("directions", cfg["evaluation"]["steered_directions"]))
    signs = set(b.get("signs", [1, -1]))
    strs = set(b.get("strengths", strengths))
    out = []
    for c in eval_conditions(cfg, man, strengths):
        if c == "none":
            out.append(c)
            continue
        if c.startswith("natural_"):
            if b.get("natural", True):
                out.append(c)
            continue
        d, signed = parse_condition(c)
        if (1 if signed >= 0 else -1) not in signs:
            continue
        kind = man["directions"][d]["kind"]
        if kind == "random":
            if b.get("random", True):
                out.append(c)
        elif kind == "projection" or (d in keep_dirs and abs(signed) in strs):
            out.append(c)
    return out


def balanced_plan(jobs: dict[str, list[str]], sizes: dict[str, int], deps: dict[str, str], shards: int) -> list[dict[str, list[str]]]:
    """Assign (benchmark, condition) jobs to shards, greedy by prompt count (largest first,
    deterministic tie-break). A dependent benchmark's job travels with its source job
    (same condition), since it is built from that job's output on the same pod."""
    units: dict[tuple[str, str], list[str]] = {}
    for b, conds in jobs.items():
        if b in deps:
            continue
        for c in conds:
            units[(b, c)] = [b]
    for b, src in deps.items():
        for c in jobs.get(b, []):
            if (src, c) in units:
                units[(src, c)].append(b)
    weight = {u: sum(sizes[b] for b in bs) for u, bs in units.items()}
    load = [0] * shards
    out: list[dict[str, list[str]]] = [{} for _ in range(shards)]
    for u in sorted(units, key=lambda u: (-weight[u], u)):
        k = min(range(shards), key=lambda i: (load[i], i))
        load[k] += weight[u]
        for b in units[u]:
            out[k].setdefault(b, []).append(u[1])
    return out


def shard_of(cond: str, all_conds: list[str], shards: int) -> int:
    """Whole conditions go to one shard (round-robin over the plan's condition order)."""
    return all_conds.index(cond) % shards


def main():
    p = base_parser(__doc__)
    p.add_argument("--steering-config", default=str(ROOT / "configs" / "steering.yaml"))
    p.add_argument("--phase", choices=["sanity", "calibrate", "eval", "all"], default="all")
    p.add_argument("--strengths", nargs="+", type=float, default=None, help="override calibrated strengths")
    p.add_argument("--benchmarks", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None, help="prompts per benchmark (smoke)")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--shard", type=int, default=0, help="this pod's shard (conditions split round-robin)")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--parallel-conditions", type=int, default=3, help="conditions generated concurrently")
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

    fixed = cfg["evaluation"].get("strengths")
    if args.phase in ("calibrate", "all") and not (fixed or args.strengths):
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
        strengths = args.strengths or fixed or json.loads((sd / "calibration.json").read_text())["eval_strengths"]
        conds = eval_conditions(cfg, man, strengths)
        benches = args.benchmarks or list(cfg["evaluation"]["benchmarks"])
        bench_list = [b for b in benches if b != "persona"]
        deps = {b: (cfg["evaluation"]["benchmarks"].get(b) or {}).get("depends_on") for b in bench_list}
        deps = {b: d for b, d in deps.items() if d}
        all_jobs = {b: benchmark_conditions(cfg, man, strengths, b) for b in bench_list}
        if cfg["evaluation"].get("shard_by") == "jobs":     # stage B on: balanced by prompt count
            sizes = {b: len(benchmark_prompts(deps.get(b, b), cfg, exp)) for b in bench_list}
            plan = balanced_plan(all_jobs, sizes, deps, args.shards)[args.shard]
        else:                                                 # stage A: whole conditions round-robin
            plan = {b: [c for c in cs if shard_of(c, conds, args.shards) == args.shard] for b, cs in all_jobs.items()}
            for b in list(plan):
                if b in deps:
                    plan[b] = [c for c in plan[b] if c in plan.get(deps[b], [])]
        write_json(sd / f"eval_plan_shard{args.shard}of{args.shards}.json",
                   {"strengths": strengths, "conditions": conds, "shard": args.shard, "shards": args.shards, "plan": plan})
        print(f"[eval] shard {args.shard}/{args.shards}: " + ", ".join(f"{b} {len(c)}" for b, c in plan.items()), flush=True)
        bcfg = cfg["evaluation"]["benchmarks"]
        dependent = {b for b in plan if (bcfg.get(b) or {}).get("depends_on")}
        jobs = [(b, c) for b, bconds in plan.items() for c in bconds if b not in dependent]
        later = [(b, c) for b, bconds in plan.items() for c in bconds if b in dependent]
        import threading

        local = threading.local()

        def run_job(job):
            # one client per worker thread (an async client is bound to its event loop); several
            # conditions in flight keep the server's queue full through each batch's long tail
            b, c = job
            dep = (bcfg.get(b) or {}).get("depends_on")
            if dep:   # e.g. turn 2 of a two-turn episode, built from this condition's turn-1 answers
                from gshift.io import read_jsonl
                from gshift.steer.proxy_panel import rulechange_turn2_prompts

                prompts = rulechange_turn2_prompts(b, read_jsonl(sd / "gen" / dep / f"{c}.jsonl"), cfg)
            else:
                prompts = benchmark_prompts(b, cfg, exp, natural_prefix(c, exp))
            if args.limit:
                prompts = prompts[: args.limit]
            if not hasattr(local, "tok"):    # fast tokenizers are not safe to share across threads
                local.tok = load_tokenizer(spec)
            wc = SteerClient(args.base_url, spec.key, max(1, args.concurrency // args.parallel_conditions))
            bgen = dict(gen)
            if (cfg["evaluation"]["benchmarks"].get(b) or {}).get("max_tokens") and not args.smoke:
                bgen["max_tokens"] = int(cfg["evaluation"]["benchmarks"][b]["max_tokens"])   # e.g. agentic
            return generate_condition(wc, local.tok, spec, prompts, c, paths, man, bgen, sd / "gen" / b / f"{c}.jsonl")

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.parallel_conditions) as pool:
            for _ in pool.map(run_job, jobs):
                pass
            for _ in pool.map(run_job, later):     # after every turn-1 condition has finished
                pass
        if "persona" in benches and args.shard == 0 and (bcfg.get("persona") or {}).get("generate", True):
            ac = cfg["activation_check"]
            s_star = max(strengths)
            persona_dirs = list(ac["directions"]) + [f"random{k}" for k in range(int(ac.get("n_random", 0)))]
            for c in sorted({"none"} | {condition_name(d, sg * s_star) for d in persona_dirs for sg in (1, -1)}):
                prompts = benchmark_prompts("persona", cfg, exp)
                if args.limit:
                    prompts = prompts[: args.limit]
                generate_condition(client, tok, spec, prompts, c, paths, man, gen, sd / "gen" / "persona" / f"{c}.jsonl")

if __name__ == "__main__":
    main()
