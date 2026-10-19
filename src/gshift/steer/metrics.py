"""Model-free response metrics: refusal, length, formatting, degeneration."""

from __future__ import annotations

import re
from collections import Counter

REFUSAL_PATTERNS = [
    r"^\s*(i'?m sorry|i am sorry|sorry,)", r"\bi can(?:'|no)t (?:help|assist|comply|do that|provide)",
    r"\bi (?:won'?t|will not) (?:help|assist|provide|do)", r"\bi'?m (?:not able|unable) to (?:help|assist|provide)",
    r"\bas an ai\b.*\b(can'?t|cannot|unable)\b",
]
_REFUSAL = [re.compile(p, re.I | re.S) for p in REFUSAL_PATTERNS]


def is_refusal(text: str) -> bool:
    head = (text or "")[:400]
    return any(p.search(head) for p in _REFUSAL)


def formatting(text: str) -> dict[str, float]:
    t = text or ""
    lines = t.splitlines()
    return {
        "n_chars": len(t),
        "n_lines": len(lines),
        "n_headers": sum(1 for l in lines if re.match(r"^\s*#{1,6}\s", l)),
        "n_bullets": sum(1 for l in lines if re.match(r"^\s*([-*+]|\d+[.)])\s", l)),
        "n_bold": len(re.findall(r"\*\*[^*]+\*\*", t)),
        "n_code_blocks": t.count("```") // 2,
        "n_table_rows": sum(1 for l in lines if l.strip().startswith("|") and l.strip().endswith("|")),
        "n_emoji": len(re.findall(r"[\U0001F300-\U0001FAFF☀-➿]", t)),
    }


def repetition_rate(text: str, n: int = 4) -> float:
    """Share of word 4-grams that are repeats; > ~0.5 indicates degenerate looping."""
    w = (text or "").split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    if len(grams) < 8:
        return 0.0
    c = Counter(grams)
    return 1.0 - len(c) / len(grams)


def degenerate(text: str, threshold: float = 0.5) -> bool:
    return repetition_rate(text) > threshold


def response_metrics(rec: dict) -> dict[str, float]:
    ans = rec.get("answer_text", "")
    m = {"has_answer": float(bool(rec.get("has_answer"))), "truncated": float(bool(rec.get("truncated"))),
         "n_answer_tokens": float(rec.get("n_answer_tokens", 0)),
         "n_reasoning_tokens": float(rec.get("n_reasoning_tokens", 0)),
         "refusal": float(is_refusal(ans)) if rec.get("has_answer") else float("nan"),
         "repetition_rate": repetition_rate(ans), "degenerate": float(degenerate(ans))}
    m.update({k: float(v) for k, v in formatting(ans).items()})
    return m
