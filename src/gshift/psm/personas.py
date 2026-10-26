"""Track A persona set (frozen 2026-10-26, before any new generation).

Rule: among the 50 roles shared by the Qwen (100-role) and OLMo (50-role) designs, take, within
each of the 5 categories, the roles at the minimum, median and maximum of their projection on
the Assistant contrast (default Assistant centroid minus the mean role centroid) in the
Qwen3.6-27B Q1 landscape (open-ended questions, baseline condition, discovery questions, last
prompt token, layer 36). 15 roles + the unprompted default Assistant = 16 personas. The rule
spans the landscape without looking at any behaviour.

    python -m gshift.psm.personas <q1 centroids .npz> <experiment_olmo_rl.yaml>   # writes the file
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from .. import DATA_DIR

PERSONA_FILE = DATA_DIR / "psm" / "personas_v1.json"
DEFAULT = "default"


def select(centroids_npz: str | Path, olmo_roles: dict[str, list[str]]) -> dict:
    z = np.load(centroids_npz, allow_pickle=True)
    roles = [str(r) for r in z["roles"]]
    ci = [str(c) for c in z["conds"]].index("baseline")
    mu = z["mu_discovery"][:, ci].astype(np.float64)
    md = z["mu_default_discovery"][ci].astype(np.float64)
    axis = md - mu.mean(0)
    axis /= np.linalg.norm(axis)
    proj = {r: float((mu[i] - mu.mean(0)) @ axis) for i, r in enumerate(roles)}
    chosen, detail = [], {}
    for cat, rs in olmo_roles.items():
        rs = sorted(rs, key=lambda r: proj[r])
        picks = [rs[0], rs[len(rs) // 2], rs[-1]]
        for tag, r in zip(("min", "median", "max"), picks):
            chosen.append(r)
            detail[r] = {"category": cat, "position": tag, "assistant_axis_projection": proj[r]}
    return {"version": 1, "frozen_on": "2026-10-26", "default": DEFAULT,
            "personas": [DEFAULT] + chosen, "roles": detail,
            "rule": __doc__.split("Rule: ")[1].split("\n\n")[0].replace("\n", " "),
            "source": str(centroids_npz)}


def load_personas() -> list[str]:
    return json.loads(PERSONA_FILE.read_text())["personas"]


if __name__ == "__main__":
    import yaml

    out = select(sys.argv[1], yaml.safe_load(open(sys.argv[2]))["roles"])
    PERSONA_FILE.parent.mkdir(parents=True, exist_ok=True)
    PERSONA_FILE.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["personas"]))
