#!/usr/bin/env python3
"""Q2 step 4 (GPU, HF): does steering move held-out persona activations in the direction
observed under natural grading prompts? -> outputs/<model>/steering/activation_check.json

    python -u scripts/13_steer_activation_check.py --model qwen3.6-27b
"""

import json

from _common import ROOT, base_parser, experiment_for

from gshift.config import get_model, load_yaml
from gshift.extract import load_hf_model
from gshift.io import write_json
from gshift.steer.activation_check import run_activation_check
from gshift.steer.tasks import benchmark_prompts
from gshift.steer.vectors import load_steering_set, steering_dir


def main():
    p = base_parser(__doc__)
    p.add_argument("--steering-config", default=str(ROOT / "configs" / "steering.yaml"))
    p.add_argument("--strength", type=float, default=None, help="default: calibrated s*")
    args = p.parse_args()
    exp = experiment_for(args)
    cfg = load_yaml(args.steering_config)
    spec = get_model(args.model)
    man, vecs = load_steering_set(spec.key)
    sd = steering_dir(spec.key)
    strength = args.strength or json.loads((sd / "calibration.json").read_text())["s_star"]
    ac = cfg["activation_check"]
    dirs = [d for d in ac["directions"] if d in vecs] + [f"random{k}" for k in range(int(ac.get("n_random", 0)))
                                                         if f"random{k}" in vecs]
    prompts = benchmark_prompts("persona", cfg, exp)
    model, tok = load_hf_model(spec)
    res = run_activation_check(model, tok, spec, prompts, man, vecs, dirs, strength, list(ac["downstream_layers"]),
                               gen_dir=sd / "gen" / "persona")
    write_json(sd / "activation_check.json", res)
    print(json.dumps(res["pre"], indent=1)[:3000])


if __name__ == "__main__":
    main()
