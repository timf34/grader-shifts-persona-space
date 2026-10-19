"""Stage 3: blind role-adherence scoring (Assistant Axis rubric, 0-3).

The judge sees ONLY the original question (without any grading statement) and the final
answer text (reasoning removed). It never sees the system prompt, the condition label or
the condition statement. Anything the answer itself says about being graded is the
model's own output and is left in (reported as a caveat, not scrubbed).
"""

from __future__ import annotations

import asyncio
import os
import random
import re
from pathlib import Path
from typing import Any

from .config import load_role
from .io import append_jsonl, read_jsonl


def parse_score(text: str | None) -> int | None:
    """First integer in the reply, accepted only if it is 0..3 (Assistant Axis rule)."""
    if not text:
        return None
    m = re.search(r"\b(\d+)\b", text.strip())
    if not m:
        return None
    v = int(m.group(1))
    return v if 0 <= v <= 3 else None


def build_judge_prompt(role: str, question: str, answer: str) -> str:
    tmpl = load_role(role)["eval_prompt"]
    # literal substitution: never str.format on model output
    return tmpl.replace("{question}", question).replace("{answer}", answer)


class JudgeClient:
    def __init__(self, provider: str, model: str, max_tokens: int, temperature: float | None):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        if provider == "openai":
            import openai

            self.client = openai.AsyncOpenAI()
        elif provider == "anthropic":
            import anthropic

            self.client = anthropic.AsyncAnthropic()
        else:
            raise ValueError(f"unknown judge provider {provider!r}")

    async def __call__(self, prompt: str) -> str | None:
        if self.provider == "openai":
            kw: dict[str, Any] = dict(model=self.model, messages=[{"role": "user", "content": prompt}],
                                      max_completion_tokens=self.max_tokens)
            if self.temperature is not None:
                kw["temperature"] = self.temperature
            r = await self.client.chat.completions.create(**kw)
            return r.choices[0].message.content if r.choices else None
        r = await self.client.messages.create(
            model=self.model, max_tokens=max(self.max_tokens, 16),
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


async def _score_all(client: JudgeClient, items: list[dict], concurrency: int, max_attempts: int = 5) -> None:
    from collections import defaultdict

    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    pending: dict[str, list[dict]] = defaultdict(list)
    n_pending = 0

    def flush():
        nonlocal n_pending
        for path, rows in pending.items():
            if rows:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                append_jsonl(path, rows)
        pending.clear()
        n_pending = 0

    async def one(item):
        nonlocal n_pending
        async with sem:
            reply, err = None, None
            for attempt in range(max_attempts):
                try:
                    reply = await client(item["prompt"])
                    if parse_score(reply) is not None:
                        break
                except Exception as e:  # rate limits, 5xx: back off and retry
                    err = repr(e)
                    await asyncio.sleep(min(60, 2 ** attempt + random.random()))
            row = {"id": item["id"], "role": item["role"], "condition": item["condition"],
                   "question_id": item["question_id"], "score": parse_score(reply), "reply": reply,
                   "error": err if reply is None else None, "judge_provider": client.provider,
                   "judge_model": client.model}
            async with lock:
                pending[item["out"]].append(row)
                n_pending += 1
                if n_pending >= 200:
                    flush()

    await asyncio.gather(*(one(it) for it in items))
    flush()


def run_judge(model_key: str, pairs: list[tuple[str, str]], judge_cfg: dict[str, Any],
              default_role: str = "default", limit: int | None = None) -> int:
    """Score every (condition, role) response file in `pairs` -> scores/<role>.jsonl.
    The default-Assistant control has no role rubric and is not judged."""
    from .io import responses_path, scores_path

    items, n_skip = [], 0
    for cond, role in pairs:
        out = scores_path(model_key, cond, role)
        done = {r["id"] for r in read_jsonl(out) if r.get("score") is not None or r.get("skipped")}
        skipped = []
        for rec in read_jsonl(responses_path(model_key, cond, role)):
            if rec["id"] in done:
                continue
            base = {"id": rec["id"], "role": role, "condition": cond, "question_id": rec["question_id"]}
            if role == default_role:
                skipped.append({**base, "score": None, "skipped": "default_role"})
            elif not rec.get("has_answer"):
                skipped.append({**base, "score": None, "skipped": "no_answer"})
            else:
                # ORIGINAL question (no prefix, no condition label) + final answer only
                items.append({**base, "out": str(out),
                              "prompt": build_judge_prompt(role, rec["question"], rec["answer_text"])})
        if skipped:
            out.parent.mkdir(parents=True, exist_ok=True)
            append_jsonl(out, skipped)
            n_skip += len(skipped)
    if limit is not None:
        items = items[:limit]
    print(f"[judge] {len(items)} to score, {n_skip} skipped (default role / no answer)", flush=True)
    if not items:
        return 0
    key_env = "OPENAI_API_KEY" if judge_cfg["provider"] == "openai" else "ANTHROPIC_API_KEY"
    if not os.environ.get(key_env):
        raise RuntimeError(f"{key_env} is not set (put it in .env)")
    client = JudgeClient(judge_cfg["provider"], judge_cfg["model"], int(judge_cfg.get("max_tokens", 10)),
                         judge_cfg.get("temperature"))
    asyncio.run(_score_all(client, items, int(judge_cfg.get("concurrency", 32))))
    return len(items)


def latest_scores(path: Path) -> dict[str, dict]:
    """id -> last row (later rows supersede earlier failed attempts)."""
    out: dict[str, dict] = {}
    for r in read_jsonl(path):
        out[r["id"]] = r
    return out
