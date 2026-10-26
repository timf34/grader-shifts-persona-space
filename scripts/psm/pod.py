#!/usr/bin/env python3
"""PSM experiment, pod side. Phases (run in this order by scripts/pod/run_psm.sh):

  hf       HF forward passes (no server): vignette features -> trait directions per layer
           (+ held-out validation), and last-prompt-token residuals of every unique report
           prompt of the stage (persona landscape on the assay; trait projection pre-decision).
  vectors  unit directions per layer (trait, random<k>, traitdpo if available) -> GGUF + manifest
           with A_L (Betley-convention activation norm) for the steering scale.
  gen      EasySteer client: all jobs of the stage (resumable per prompt and condition).
  select   pre-registered (layer, strength) choice from the pilot sweep (configs/psm.yaml).

Outputs: outputs/<model>/psm/{hf,vectors,gguf,gen,...}.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from gshift.config import get_model  # noqa: E402
from gshift.io import model_dir, read_jsonl, write_json  # noqa: E402
from gshift.psm import plan as P  # noqa: E402


def psm_dir(model_key: str) -> Path:
    d = model_dir(model_key) / "psm"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# hf
# ---------------------------------------------------------------------------

def phase_hf(spec, cfg, stage: str):
    from gshift.extract import Extractor, load_hf_model
    from gshift.psm.vignettes import load_vignettes, trait_directions, vignette_features
    from gshift.templates import prompt_token_ids

    out = psm_dir(spec.key) / "hf"
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in cfg["hf"]["layers"]]
    model, tok = load_hf_model(spec)
    ex = Extractor(model, tok, layers, True, spec.chat_template_kwargs)
    t0 = time.time()
    vf = out / "vignette_feats.npz"
    vig = load_vignettes(cfg["vignettes_file"])
    if not vf.exists():
        feats = vignette_features(ex, tok, vig, spec.chat_template_kwargs)
        np.savez(vf, layers=np.array(layers), ids=np.array([v["id"] for v in vig]),
                 splits=np.array([v["split"] for v in vig]), **feats)
        print(f"[hf] vignettes: {len(vig)} pairs in {time.time() - t0:.0f}s", flush=True)
    z = np.load(vf)
    td = {}
    for rep in ("ending", "last"):
        r = trait_directions({k: z[k] for k in z.files}, [str(s) for s in z["splits"]], layers, rep)
        td[rep] = r
        np.savez(out / f"trait_dirs_{rep}.npz", **{f"L{L}": u for L, u in r["dirs"].items()})
    write_json(out / "trait_stats.json", {rep: {str(L): s for L, s in td[rep]["stats"].items()} for rep in td})
    print("[hf] trait (ending) held-out sign acc by layer: " +
          ", ".join(f"L{L} {s['heldout_pair_sign_acc']:.2f}/d={s['heldout_d']:.2f}" for L, s in td["ending"]["stats"].items()),
          flush=True)

    af = out / f"acts_{stage}.npz"
    if not af.exists():
        prompts = P.activation_prompts(cfg, stage)
        save_layers = [int(x) for x in cfg["hf"]["save_acts_layers"]]
        li = [layers.index(L) for L in save_layers]
        t1 = time.time()
        A = np.zeros((len(prompts), len(save_layers), ex.prompt_last(prompt_token_ids(
            tok, prompts[0].messages, True, spec.chat_template_kwargs)).shape[-1]), np.float16)
        for i, p in enumerate(prompts):
            A[i] = ex.prompt_last(prompt_token_ids(tok, p.messages, True, spec.chat_template_kwargs))[li]
            if i % 500 == 0:
                print(f"[hf] acts {i}/{len(prompts)} ({time.time() - t1:.0f}s)", flush=True)
        np.savez(af, acts=A, layers=np.array(save_layers), bench=np.array([p.benchmark for p in prompts]),
                 ids=np.array([p.id for p in prompts]),
                 meta=np.array([json.dumps(p.meta) for p in prompts]))
        print(f"[hf] acts_{stage}: {A.shape} in {time.time() - t1:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# vectors
# ---------------------------------------------------------------------------

def act_norms(spec) -> list[float]:
    rep = model_dir(spec.key) / "betley_repro" / "report.json"
    return json.loads(rep.read_text())["per_layer_mean_activation_norm"]


def phase_vectors(spec, cfg, transfer_from: str | None = None):
    from gshift.steer.engine import write_direction_gguf

    d = psm_dir(spec.key)
    vd, gd = d / "vectors", d / "gguf"
    vd.mkdir(exist_ok=True)
    tz = np.load(d / "hf" / "trait_dirs_ending.npz")
    norms = act_norms(spec)
    man = {"model_key": spec.key, "scale": "strength x A_L x unit (A_L = betley_repro mean residual norm)",
           "A_L": {}, "directions": {}}
    sw = cfg["steering"]["sweep"]
    layers = sorted({int(x) for x in sw["layers"]} | {36})
    n_rand = max(int(sw["random"]["n"]), int(cfg["confirm"]["track_b"]["random"]["n"]))
    H = None
    for L in layers:
        man["A_L"][str(L)] = float(norms[L])
        vecs = {"trait": tz[f"L{L}"].astype(np.float32)}
        H = vecs["trait"].size
        for k in range(n_rand):
            r = np.random.default_rng([int(cfg["steering"]["random_seed"]), k, L]).normal(size=H)
            vecs[f"random{k}"] = (r / np.linalg.norm(r)).astype(np.float32)
        if transfer_from:
            src = model_dir(transfer_from) / "psm" / "hf" / "trait_dirs_ending.npz"
            if src.exists():
                vecs["traitdpo"] = np.load(src)[f"L{L}"].astype(np.float32)
        for name, u in vecs.items():
            key = f"{name}@L{L}"
            np.save(vd / f"{key}.npy", u)
            write_direction_gguf(u, L, gd / f"{key}.gguf")
            man["directions"][key] = {"layer": L, "cos_trait": float(u @ vecs["trait"])}
    write_json(d / "vectors.json", man)
    print(f"[vectors] {len(man['directions'])} directions at layers {layers}; A_L {man['A_L']}", flush=True)


# ---------------------------------------------------------------------------
# gen
# ---------------------------------------------------------------------------

def phase_gen(spec, cfg, stage: str, base_url: str, concurrency: int, parallel: int, limit: int | None,
              transfer: bool, only: list[str] | None):
    from gshift.generate import load_tokenizer
    from gshift.steer.engine import SteerClient, generate_condition, sanity_check

    d = psm_dir(spec.key)
    man = json.loads((d / "vectors.json").read_text())
    gen = P.gen_cfg(cfg, spec.key)
    choice = None
    if stage == "confirm":
        choice = json.loads((d / "steer_choice.json").read_text())["choice"]
    jobs = P.stage_jobs(cfg, stage, choice, transfer and any(k.startswith("traitdpo@") for k in man["directions"]))
    if only:
        jobs = [j for j in jobs if any(j.bench.startswith(o) for o in only)]
    tok = load_tokenizer(spec)
    client = SteerClient(base_url, spec.key, concurrency)
    sres = sanity_check(client, tok, spec, d / "gguf" / "trait@L36.gguf", 36, float(man["A_L"]["36"]))
    write_json(d / "sanity.json", sres)
    print("[sanity]", json.dumps({k: v for k, v in sres.items() if not k.endswith("_text")}), flush=True)
    if not sres["ok"]:
        sys.exit("steering sanity check failed")
    local = threading.local()
    timing = []
    lock = threading.Lock()

    def run(job: P.Job):
        prompts = job.prompts[:limit] if limit else job.prompts
        if not hasattr(local, "tok"):
            local.tok = load_tokenizer(spec)
        direction, L, s = P.parse_cond(job.condition)
        cmanifest = {"layer": L or 36, "activation_norm": float(man["A_L"][str(L or 36)]), "directions": {}}
        paths = {}
        cond = job.condition
        if direction:
            key = f"{direction}@L{L}"
            if key not in man["directions"]:
                print(f"[gen] skip {job.bench}/{cond}: direction {key} not built", flush=True)
                return 0
            cmanifest["directions"][key] = {"norm_factor": 1.0}
            paths[key] = d / "gguf" / f"{key}.gguf"
        wc = SteerClient(base_url, spec.key, max(1, concurrency // parallel))
        t0 = time.time()
        n = generate_condition(wc, local.tok, spec, prompts, cond, paths, cmanifest, gen,
                               d / "gen" / job.bench / f"{cond}.jsonl")
        if n:
            rows = read_jsonl(d / "gen" / job.bench / f"{cond}.jsonl")
            ntok = sum(len(r["gen_ids"]) for r in rows[-n:])
            with lock:
                timing.append({"bench": job.bench, "condition": cond, "n": n, "seconds": time.time() - t0,
                               "gen_tokens": ntok})
                write_json(d / f"timing_{stage}.json", timing)
        return n

    order = sorted(jobs, key=lambda j: (not j.bench.startswith(("rpt_", "steer_base")), j.bench, j.condition))
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        total = sum(pool.map(run, order))
    print(f"[gen] {stage}: {total} new generations", flush=True)


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------

def phase_select(spec, cfg):
    from gshift.psm.score import select_steering

    d = psm_dir(spec.key)
    res = select_steering(d / "gen" / "steer_base", cfg)
    write_json(d / "steer_choice.json", res)
    print("[select]", json.dumps({k: res[k] for k in ("choice", "gate_passed", "reason")}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["hf", "vectors", "gen", "select"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--stage", choices=["pilot", "confirm"], default="pilot")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--concurrency", type=int, default=192)
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--transfer-from", default=None, help="model key whose trait directions to add as traitdpo")
    ap.add_argument("--only", nargs="+", default=None, help="benchmark prefixes to run")
    a = ap.parse_args()
    spec = get_model(a.model)
    cfg = P.load_cfg()
    if a.phase == "hf":
        phase_hf(spec, cfg, a.stage)
    elif a.phase == "vectors":
        phase_vectors(spec, cfg, a.transfer_from)
    elif a.phase == "gen":
        phase_gen(spec, cfg, a.stage, a.base_url, a.concurrency, a.parallel, a.limit, bool(a.transfer_from), a.only)
    else:
        phase_select(spec, cfg)


if __name__ == "__main__":
    main()
