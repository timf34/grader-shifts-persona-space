#!/usr/bin/env python3
"""Track B extraction data: third-person character vignettes, each with a shared setup and two
matched endings (the character states the facts as they are / knowingly misrepresents them).

    python scripts/psm/make_vignettes.py            # -> data/psm/vignettes_v1.jsonl (+ raw/)

Written by an LLM through OpenRouter (the only LLM provider used in this project), then
filtered by fixed rules (banned trait words, length match, duplicates) and split 75/25 into
train / held-out with a fixed seed. Independence from the decision assay: no scene may involve
reporting a number, count, measurement, test result or inspection outcome (the assay's
structure), and endings describe only the act, never its consequences.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "psm" / "vignettes_v1.jsonl"
RAW = ROOT / "data" / "psm" / "raw"
MODEL = "anthropic/claude-sonnet-5.5"

BANNED = ("honest", "dishonest", "lie", "lied", "lies", "lying", "liar", "truth", "untrue", "deceiv", "decept",
          "deceit", "cheat", "fraud", "integrity", "sincer", "mislead", "misled", "fib", "trust", "fake", "false",
          "pretend", "made up", "make up", "fabricat", "invent", "conceal", "cover up", "covered up", "admit",
          "confess", "candid", "frank", "virtue", "moral", "ethic", "right thing", "wrong thing", "guilt")
# the assay's structure (numbers in official reports) must not appear in extraction scenes
ASSAY_WORDS = ("report", "count", "inventory", "audit", "inspection", "inspector", "test result", "metric",
               "measurement", "survey", "emission", "quota", "target")

DOMAINS = [
    "buying and selling second-hand things between private people",
    "friends, flatmates and neighbours",
    "family life and relatives",
    "school, university and hobbies (not exams or grades)",
    "small shops, restaurants and services (customer-facing conversations)",
    "travel, rentals and borrowing things",
    "workplace conversations between colleagues (no reports, numbers or metrics)",
]
N_PER_BATCH = 20

PROMPT = """Write {n} short third-person vignettes for a psychology study of how people describe other people's conduct. Domain for this batch: {domain}.

Each vignette has a SETUP and two alternative ENDINGS.
- SETUP: 2-4 sentences. A named character (vary names, genders, ages and cultural backgrounds; never reuse a name in this batch) knows a specific fact that matters to another person, and has a reason to want the other person to believe something different. End the setup right before the character speaks or acts. Do not describe the character's personality or intentions.
- ENDING_A: 1-2 sentences. The character tells the other person the fact as it is.
- ENDING_B: 1-2 sentences. The character knowingly tells the other person something that contradicts the fact, or presents it so the other person will believe the opposite, to get the benefit or avoid the cost.
- Both endings: describe only what the character says or does, in concrete words (include a short line of what they say). Never describe consequences, reactions, feelings, or whether it worked. The two endings must have about the same length (within 20% in words) and similar sentence structure.
- Never use these words or their variants anywhere: honest, dishonest, lie, liar, truth, true, false, deceive, deception, cheat, fraud, integrity, sincere, mislead, fib, trust, fake, pretend, make up, invent, fabricate, conceal, cover up, admit, confess, candid, frank, moral, ethical, guilt, right thing, wrong thing.
- Never involve numbers in official documents, reports, counts, inventories, audits, inspections, tests, measurements, surveys, quotas or targets.
- Vary the situations widely; each vignette must be a different kind of situation.

Return ONLY a JSON array of {n} objects with keys "setup", "ending_a", "ending_b". No commentary."""


def bad_words(text: str) -> list[str]:
    t = " " + text.lower() + " "
    hits = [b for b in BANNED if re.search(r"(?<![a-z])" + re.escape(b), t)]
    hits += [w for w in ASSAY_WORDS if re.search(r"(?<![a-z])" + re.escape(w), t)]
    return hits


async def batch(client, domain: str, k: int, sem) -> list[dict]:
    raw = RAW / f"batch_{DOMAINS.index(domain)}_{k}.json"
    if raw.exists():
        return json.loads(raw.read_text())
    async with sem:
        for attempt in range(4):
            try:
                r = await client.chat.completions.create(
                    model=MODEL, max_tokens=8000, temperature=1.0,
                    messages=[{"role": "user", "content": PROMPT.format(n=N_PER_BATCH, domain=domain)}],
                    extra_body={"provider": {"order": ["anthropic"], "allow_fallbacks": False}})
                txt = r.choices[0].message.content.strip()
                txt = txt[txt.index("["): txt.rindex("]") + 1]
                rows = json.loads(txt)
                RAW.mkdir(parents=True, exist_ok=True)
                raw.write_text(json.dumps(rows, indent=1))
                return rows
            except Exception as e:  # noqa: BLE001
                print(f"[vignettes] {domain} #{k} attempt {attempt}: {e}", flush=True)
                await asyncio.sleep(5 * (attempt + 1))
    return []


async def main_async(n_batches: int) -> list[dict]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
    sem = asyncio.Semaphore(8)
    jobs = [batch(client, d, k, sem) for d in DOMAINS for k in range(n_batches)]
    res = await asyncio.gather(*jobs)
    out = []
    for (d, k), rows in zip([(d, k) for d in DOMAINS for k in range(n_batches)], res):
        for j, r in enumerate(rows):
            out.append({"domain": d, "batch": k, "index": j, **r})
    return out


def main():
    n_batches = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    rows = asyncio.run(main_async(n_batches))
    kept, dropped = [], {}
    seen = set()
    for r in rows:
        s, a, b = (str(r.get(k, "")).strip() for k in ("setup", "ending_a", "ending_b"))
        if not (s and a and b):
            dropped["missing"] = dropped.get("missing", 0) + 1
            continue
        hits = bad_words(" ".join([s, a, b]))
        if hits:
            dropped["banned:" + hits[0]] = dropped.get("banned:" + hits[0], 0) + 1
            continue
        la, lb = len(a.split()), len(b.split())
        if max(la, lb) > 1.35 * min(la, lb):
            dropped["length"] = dropped.get("length", 0) + 1
            continue
        key = s[:60].lower()
        if key in seen:
            dropped["duplicate"] = dropped.get("duplicate", 0) + 1
            continue
        seen.add(key)
        vid = "v" + hashlib.sha256(s.encode()).hexdigest()[:8]
        kept.append({"id": vid, "domain": r["domain"], "setup": s, "accurate": a, "misrepresent": b})
    rng = np.random.default_rng(20261026)
    order = rng.permutation(len(kept))
    n_ho = len(kept) // 4
    for rank, i in enumerate(order):
        kept[i]["split"] = "heldout" if rank < n_ho else "train"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"[vignettes] {len(rows)} written by {MODEL}, kept {len(kept)} "
          f"({sum(r['split'] == 'train' for r in kept)} train / {n_ho} held-out); dropped {dropped}")


if __name__ == "__main__":
    main()
