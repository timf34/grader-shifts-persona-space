#!/usr/bin/env python3
"""GPU-side validation (docs/gpu_validation.md). Each subcommand prints a summary, writes
JSON under <outputs root>/<model>/gpu_checks/, and exits non-zero when its check fails.

    python scripts/gpu_checks.py spans  --model qwen3.6-27b --smoke          # CPU + tokenizer
    python scripts/gpu_checks.py replay --model qwen3.6-27b --smoke          # GPU (HF)
    bash scripts/pod/serve_easysteer.sh qwen3.6-27b
    python scripts/gpu_checks.py steer-server --model qwen3.6-27b --smoke    # client of the server
    bash scripts/pod/serve_easysteer.sh stop
    python scripts/gpu_checks.py steer-hf --model qwen3.6-27b --smoke        # GPU (HF)

`--smoke` reads outputs_smoke/, `--thinking on|off` a pilot arm, neither the main grid.
The steering checks take A_L and a realistic direction (layer-L Betley delta) from the
Betley reproduction under the same root (scripts/02_extract.py writes it).
"""

import json
import sys

import numpy as np

from _common import base_parser, experiment_for

from gshift.config import get_model
from gshift.io import conditions_present, model_dir, read_jsonl, responses_path, roles_present, write_json

PROMPTS = ["Name three primary colours.", "Write one sentence about the sea.",
           "What is 17 times 23? Answer with the number only."]


def records(model: str, n: int | None = None, seed: int = 0) -> list[dict]:
    rows = [r for c in conditions_present(model) for role in roles_present(model, c)
            for r in read_jsonl(responses_path(model, c, role))]
    if n is not None and len(rows) > n:   # spread over generation lengths
        rows = sorted(rows, key=lambda r: (len(r["gen_ids"]), r["id"]))
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, n).round().astype(int)]
    return rows


def out_dir(model: str):
    d = model_dir(model) / "gpu_checks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def finish(name: str, model: str, res: dict) -> None:
    write_json(out_dir(model) / f"{name}.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("examples", "cases", "worst_records")}, indent=1))
    print(f"[gpu_checks] {name}: {'OK' if res['ok'] else 'FAILED'} -> {out_dir(model) / (name + '.json')}")
    sys.exit(0 if res["ok"] else 1)


def betley_inputs(model: str, layer: int, hidden: int) -> tuple[float, np.ndarray, str]:
    d = model_dir(model) / "betley_repro"
    rep = json.loads((d / "report.json").read_text())
    a = float(rep["per_layer_mean_activation_norm"][layer])
    deltas = d / "deltas_all_layers.npy"
    if deltas.exists():
        v = np.load(deltas)[layer].astype(np.float64)
        src = "betley_repro delta at layer L"
    else:
        v = np.random.default_rng(0).normal(size=hidden)
        src = "random (seed 0)"
    return a, (v / np.linalg.norm(v)).astype(np.float32), src


def main():
    p = base_parser(__doc__)
    p.add_argument("check", choices=["spans", "replay", "steer-server", "steer-hf"])
    p.add_argument("--n", type=int, default=16, help="records for replay")
    p.add_argument("--layer", type=int, default=36)
    p.add_argument("--strength", type=float, default=0.5, help="steering strength in units of A_L")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    args = p.parse_args()
    exp = experiment_for(args)
    spec = get_model(args.model)
    gen = exp["generation"]
    thinking = bool(gen["enable_thinking"])

    if args.check == "spans":
        from gshift.generate import load_tokenizer
        from gshift.templates import inspect_template
        from gshift.validation import check_spans

        tok = load_tokenizer(spec)
        info = inspect_template(tok, thinking, spec.chat_template_kwargs)
        rows = records(spec.key)
        if not rows:
            sys.exit(f"no generations for {spec.key} under {model_dir(spec.key)}")
        finish("spans", spec.key, check_spans(rows, tok, info))

    if args.check == "replay":
        from gshift.extract import load_hf_model
        from gshift.validation import replay_check

        model, _ = load_hf_model(spec, exp["extraction"].get("dtype", "bfloat16"))
        rows = records(spec.key, args.n)
        finish("replay", spec.key, replay_check(model, rows, top_k=int(gen["top_k"])))

    L = args.layer
    if args.check == "steer-server":
        from openai import OpenAI

        from gshift.generate import load_tokenizer
        from gshift.steer.engine import steering_spec, write_direction_gguf
        from gshift.templates import prompt_token_ids

        a_l, unit, src = betley_inputs(spec.key, L, int(spec.hidden_size))
        gguf = write_direction_gguf(unit, L, out_dir(spec.key) / f"direction_L{L}.gguf")
        np.save(out_dir(spec.key) / f"direction_L{L}.npy", unit)
        tok = load_tokenizer(spec)
        cl = OpenAI(base_url=args.base_url, api_key="-", timeout=600)

        def call(ids, steering, sample: bool, seed: int = 0):
            body = {"skip_special_tokens": False, "return_token_ids": True,
                    "top_k": int(gen["top_k"]) if sample else -1, "min_p": 0.0}
            if steering:
                body["steering"] = steering
            r = cl.completions.create(model=spec.key, prompt=ids, max_tokens=32, logprobs=1, seed=seed,
                                      temperature=float(gen["temperature"]) if sample else 0.0,
                                      top_p=float(gen["top_p"]) if sample else 1.0, extra_body=body)
            c = r.choices[0]
            t = getattr(c, "token_ids", None)
            if t is None and getattr(c, "model_extra", None):
                t = c.model_extra.get("token_ids")
            return {"gen_ids": list(t) if t is not None else None, "text": c.text,
                    "token_logprobs": list(c.logprobs.token_logprobs) if c.logprobs else None}

        cases, zero = [], []
        for q in PROMPTS:
            ids = prompt_token_ids(tok, [{"role": "user", "content": q}], True, spec.chat_template_kwargs)
            base = call(ids, None, False)
            z = call(ids, steering_spec(gguf, L, 0.0), False)
            sb = call(ids, None, True, seed=123)
            sz = call(ids, steering_spec(gguf, L, 0.0), True, seed=123)
            zero.append({"prompt": q, "greedy_equal": base["gen_ids"] == z["gen_ids"],
                         "sampled_equal": sb["gen_ids"] == sz["gen_ids"]})
            cases.append({"prompt": q, "prompt_ids": ids, "signed_scale": 0.0, **base})
            for sgn in (1, -1):
                s = sgn * args.strength * a_l
                cases.append({"prompt": q, "prompt_ids": ids, "signed_scale": s, **call(ids, steering_spec(gguf, L, s), False)})
        res = {"layer_ours": L, "layer_easysteer": L - 1, "activation_norm": a_l, "strength": args.strength,
               "direction": src, "gguf": str(gguf), "zero_strength": zero, "cases": cases,
               "token_ids_returned": all(c["gen_ids"] is not None for c in cases),
               "logprobs_returned": all(c["token_logprobs"] is not None for c in cases)}
        unsteered = {c["prompt"]: c["gen_ids"] for c in cases if c["signed_scale"] == 0}
        res["steering_changes_greedy_output"] = [c["gen_ids"] != unsteered[c["prompt"]] for c in cases
                                                 if c["signed_scale"]]
        res["ok"] = (res["token_ids_returned"] and res["logprobs_returned"]
                     and all(z["greedy_equal"] and z["sampled_equal"] for z in zero)
                     and any(res["steering_changes_greedy_output"]))
        finish("steer_server", spec.key, res)

    if args.check == "steer-hf":
        from gshift.extract import load_hf_model, text_config
        from gshift.validation import compare_steering, gguf_roundtrip

        srv = json.loads((out_dir(spec.key) / "steer_server.json").read_text())
        unit = np.load(out_dir(spec.key) / f"direction_L{L}.npy")
        model, _ = load_hf_model(spec, exp["extraction"].get("dtype", "bfloat16"))
        cmp = compare_steering(model, srv["cases"], unit, L, text_config(model.config).num_hidden_layers)
        gg = gguf_roundtrip(srv["gguf"], unit, L)
        res = {"zero_strength_matches_unsteered": all(z["greedy_equal"] and z["sampled_equal"]
                                                      for z in srv["zero_strength"]),
               "steering_matches_hf_at_layer_L_scale_sign": cmp["ok"], "best_counts": cmp["best_counts"],
               "gguf_roundtrip": gg, "cases": cmp["cases"], "layer_ours": L, "layer_easysteer": L - 1,
               "note": "mean |logprob(server) - logprob(HF variant)| over the server's greedy tokens; "
                       "the intended variant 'L' (our layer, x1 scale, same sign) must be closest"}
        res["ok"] = res["zero_strength_matches_unsteered"] and cmp["ok"] and gg["ok"]
        finish("steer_hf", spec.key, res)


if __name__ == "__main__":
    main()
