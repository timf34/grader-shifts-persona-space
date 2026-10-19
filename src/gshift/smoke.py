"""Technical smoke-test grid: the same code paths on ~0.5% of the design."""

from __future__ import annotations

import copy

# 4 roles (3 discovery from different categories, 1 held out) + default, prompt variants
# 0-1, 4 questions covering original/checkable x discovery/validation (~190 generations).
SMOKE_ROLES = ["librarian", "therapist", "negotiator", "robot"]
SMOKE_QUESTIONS = ["orig_aa013", "orig_aa045", "chk_arith_compute_01", "chk_code_strings_02"]


def smoke_experiment(exp: dict) -> dict:
    e = copy.deepcopy(exp)
    keep = set(SMOKE_ROLES)
    e["roles"] = {c: [r for r in rs if r in keep] for c, rs in e["roles"].items()}
    e["roles"] = {c: rs for c, rs in e["roles"].items() if rs}
    e["role_split"]["heldout"] = ["robot"]
    e["question_filter"] = list(SMOKE_QUESTIONS)
    for st in e["stages"].values():
        st["prompt_variants"] = [v for v in st["prompt_variants"] if v in (0, 1)] or [0]
    e["prompt_variants"] = [0, 1]
    e["analysis"].update(n_permutations=50, n_permutations_sweep=20, n_bootstrap=20, n_split_halves=3)
    e["judge"]["min_matched_pairs_per_role"] = 2
    return e
