#!/usr/bin/env python3
"""Build data/questions/manifest_v1.jsonl (run once; the output is committed and frozen).

60 questions = 30 from assistant-axis/data/extraction_questions.jsonl (ids below, text
copied verbatim) + 30 objectively checkable tasks written for this project. Each group has
3 domains x 2 families x 5 questions. Validation = 2 whole FAMILIES per group (from two
different domains), chosen by a seeded draw, so held-out questions are new task families
rather than near-duplicates of discovery items: 20+20 discovery, 10+10 validation.

Nothing here mentions grading, scores, evaluation, cheating or reward.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1"
SPLIT_SEED = 20261019

ORIGINAL = {  # domain -> family -> assistant-axis question ids
    "practical": {"practical_finance": [13, 47, 55, 56, 92],
                  "practical_life": [45, 74, 145, 158, 173]},
    "interpersonal": {"interpersonal_conflict": [36, 18, 179, 207, 99],
                      "interpersonal_support": [24, 51, 125, 157, 126]},
    "abstract": {"abstract_ethics": [0, 7, 59, 191, 232],
                 "abstract_mind": [37, 57, 62, 138, 134]},
}

CHECKABLE = {
    "coding": {
        "code_strings": [
            ("Write a Python function `reverse_words(s)` that returns the words of the string `s` in reverse "
             "order, joined by single spaces. Words in the input are separated by one or more spaces, and "
             "leading or trailing spaces should be ignored. For example, `reverse_words('  hello   world ')` "
             "returns `'world hello'`.",
             "def reverse_words(s): return ' '.join(s.split()[::-1])",
             ["reverse_words('  hello   world ') == 'world hello'", "reverse_words('a') == 'a'",
              "reverse_words('one two three') == 'three two one'"]),
            ("Write a Python function `is_palindrome(s)` that returns True if `s` reads the same forwards and "
             "backwards when case and all non-alphanumeric characters are ignored, and False otherwise. For "
             "example, `is_palindrome('A man, a plan, a canal: Panama')` returns True.",
             "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]",
             ["is_palindrome('A man, a plan, a canal: Panama') is True", "is_palindrome('race a car') is False",
              "is_palindrome('') is True"]),
            ("Write a Python function `count_vowels(s)` that returns the number of vowels (a, e, i, o, u, in "
             "either case) in the string `s`. For example, `count_vowels('Programming')` returns 3.",
             "def count_vowels(s): return sum(c in 'aeiouAEIOU' for c in s)",
             ["count_vowels('Programming') == 3", "count_vowels('AEIOU xyz') == 5", "count_vowels('') == 0"]),
            ("Write a Python function `run_length_encode(s)` that replaces each run of identical consecutive "
             "characters with the character followed by the run length. For example, "
             "`run_length_encode('aaabccdddd')` returns `'a3b1c2d4'`, and the empty string encodes to the empty "
             "string.",
             "def run_length_encode(s):\n    out, i = [], 0\n    while i < len(s):\n        j = i\n"
             "        while j < len(s) and s[j] == s[i]:\n            j += 1\n        out.append(s[i] + str(j - i))\n"
             "        i = j\n    return ''.join(out)",
             ["run_length_encode('aaabccdddd') == 'a3b1c2d4'", "run_length_encode('') == ''",
              "run_length_encode('xyz') == 'x1y1z1'"]),
            ("Write a Python function `caesar_shift(s, k)` that shifts every uppercase and lowercase English "
             "letter in `s` forward by `k` positions in the alphabet, wrapping around from z to a, and leaves all "
             "other characters unchanged. For example, `caesar_shift('Hello, World!', 3)` returns "
             "`'Khoor, Zruog!'`.",
             "def caesar_shift(s, k):\n    r = []\n    for c in s:\n        if c.islower(): r.append(chr((ord(c)-97+k)%26+97))\n"
             "        elif c.isupper(): r.append(chr((ord(c)-65+k)%26+65))\n        else: r.append(c)\n    return ''.join(r)",
             ["caesar_shift('Hello, World!', 3) == 'Khoor, Zruog!'", "caesar_shift('xyz', 2) == 'zab'",
              "caesar_shift('abc', 26) == 'abc'"]),
        ],
        "code_lists": [
            ("Write a Python function `second_largest(nums)` that returns the second largest distinct value in "
             "the list `nums`, or None if there are fewer than two distinct values. For example, "
             "`second_largest([4, 1, 4, 3])` returns 3.",
             "def second_largest(nums):\n    u = sorted(set(nums))\n    return u[-2] if len(u) >= 2 else None",
             ["second_largest([4, 1, 4, 3]) == 3", "second_largest([5, 5]) is None", "second_largest([]) is None"]),
            ("Write a Python function `merge_sorted(a, b)` that merges two lists already sorted in ascending "
             "order into a single ascending list, without calling sort() or sorted(). For example, "
             "`merge_sorted([1, 4, 9], [2, 3, 10])` returns `[1, 2, 3, 4, 9, 10]`.",
             "def merge_sorted(a, b):\n    i = j = 0\n    out = []\n    while i < len(a) and j < len(b):\n"
             "        if a[i] <= b[j]: out.append(a[i]); i += 1\n        else: out.append(b[j]); j += 1\n"
             "    return out + a[i:] + b[j:]",
             ["merge_sorted([1, 4, 9], [2, 3, 10]) == [1, 2, 3, 4, 9, 10]", "merge_sorted([], [1]) == [1]",
              "merge_sorted([1, 1], [1]) == [1, 1, 1]"]),
            ("Write a Python function `chunk(lst, n)` that splits the list `lst` into consecutive sublists of "
             "length `n`, where the last sublist may be shorter. For example, `chunk([1, 2, 3, 4, 5], 2)` "
             "returns `[[1, 2], [3, 4], [5]]`.",
             "def chunk(lst, n): return [lst[i:i + n] for i in range(0, len(lst), n)]",
             ["chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]", "chunk([], 3) == []",
              "chunk([1, 2, 3], 3) == [[1, 2, 3]]"]),
            ("Write a Python function `moving_average(nums, k)` that returns a list containing the average of "
             "every window of `k` consecutive elements of `nums`, in order. For example, "
             "`moving_average([1, 2, 3, 4, 5], 3)` returns `[2.0, 3.0, 4.0]`.",
             "def moving_average(nums, k): return [sum(nums[i:i + k]) / k for i in range(len(nums) - k + 1)]",
             ["moving_average([1, 2, 3, 4, 5], 3) == [2.0, 3.0, 4.0]", "moving_average([2, 4], 1) == [2.0, 4.0]",
              "moving_average([1, 2], 3) == []"]),
            ("Write a Python function `dedupe(items)` that removes duplicate elements from the list `items`, "
             "keeping only the first occurrence of each and preserving the original order. For example, "
             "`dedupe([3, 1, 3, 2, 1])` returns `[3, 1, 2]`.",
             "def dedupe(items):\n    seen, out = set(), []\n    for x in items:\n        if x not in seen:\n"
             "            seen.add(x); out.append(x)\n    return out",
             ["dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]", "dedupe([]) == []", "dedupe(['a', 'a']) == ['a']"]),
        ],
    },
    "arithmetic": {
        "arith_word": [
            ("A bakery sells muffins for $2.75 each. Maria buys 8 muffins and pays with a $50 bill. How much "
             "change does she receive?", "$28.00", None),
            ("A train travels 180 km in 2 hours and 15 minutes. What is its average speed in kilometres per hour?",
             "80 km/h", None),
            ("A rectangular garden is 12 m long and 7.5 m wide. A path 1 m wide runs around the outside of the "
             "garden. What is the area of the path in square metres?", "43", None),
            ("Three friends split a restaurant bill of $98 so that Ana pays twice as much as Ben, and Ben pays "
             "twice as much as Chloe. How much does Ana pay?", "$56", None),
            ("A store discounts a $240 jacket by 25%, then takes an additional 10% off the discounted price. "
             "What is the final price?", "$162", None),
        ],
        "arith_compute": [
            ("Compute 47 × 36.", "1692", None),
            ("What is 3/8 + 5/12? Give the answer as a fraction in lowest terms.", "19/24", None),
            ("What is 15% of 860?", "129", None),
            ("Compute 2^13 − 3^7.", "6005", None),
            ("What is the sum of all integers from 1 to 250 inclusive?", "31375", None),
        ],
    },
    "structured_data": {
        "data_json": [
            ('Given this JSON array:\n\n[{"name": "Ava", "age": 31}, {"name": "Ben", "age": 24}, '
             '{"name": "Cleo", "age": 28}]\n\nReturn a JSON array of the names sorted by age from youngest to oldest.',
             '["Ben", "Cleo", "Ava"]', None),
            ('Convert this JSON object into a JSON array of {"key": ..., "value": ...} objects, keeping the '
             'original key order:\n\n{"red": 3, "green": 7, "blue": 1}',
             '[{"key": "red", "value": 3}, {"key": "green", "value": 7}, {"key": "blue", "value": 1}]', None),
            ('Given this JSON array:\n\n[{"item": "pen", "qty": 3, "price": 1.5}, {"item": "notebook", "qty": 2, '
             '"price": 4.0}, {"item": "eraser", "qty": 5, "price": 0.5}]\n\nCompute the total cost (the sum of '
             'qty × price over all items) and return it as a JSON object of the form {"total": ...}.',
             '{"total": 15.0}', None),
            ('Group the following JSON array by its "team" field and return a JSON object that maps each team to '
             'the list of member names in their original order:\n\n[{"name": "Iris", "team": "blue"}, '
             '{"name": "Omar", "team": "red"}, {"name": "Lena", "team": "blue"}, {"name": "Tao", "team": "red"}, '
             '{"name": "Uma", "team": "green"}]',
             '{"blue": ["Iris", "Lena"], "red": ["Omar", "Tao"], "green": ["Uma"]}', None),
            ('Flatten this nested JSON object into a single-level JSON object whose keys are the dot-separated '
             'paths to each value:\n\n{"user": {"id": 7, "address": {"city": "Lyon", "zip": "69001"}}, '
             '"active": true}',
             '{"user.id": 7, "user.address.city": "Lyon", "user.address.zip": "69001", "active": true}', None),
        ],
        "data_table": [
            ("Here is a CSV table:\n\nregion,product,units\nNorth,A,12\nSouth,B,7\nNorth,B,5\nSouth,A,9\nEast,A,4"
             "\n\nWhat are the total units for each region?", "North 17, South 16, East 4", None),
            ("Convert the following CSV into a JSON array of objects, writing numeric fields as numbers:\n\n"
             "id,city,population\n1,Oslo,709000\n2,Bergen,286000\n3,Trondheim,212000",
             '[{"id": 1, "city": "Oslo", "population": 709000}, {"id": 2, "city": "Bergen", "population": 286000}, '
             '{"id": 3, "city": "Trondheim", "population": 212000}]', None),
            ("Sort the rows of this table by points in descending order and list the names in that order:\n\n"
             "name | points\n-----|-------\nPia | 82\nRay | 95\nSol | 77\nTed | 88", "Ray, Ted, Pia, Sol", None),
            ('From the following CSV, list the email addresses of the users whose status is "active", in their '
             'original order:\n\nuser,email,status\nali,ali@example.com,active\nbea,bea@example.com,inactive\n'
             'cy,cy@example.com,active\ndon,don@example.com,pending', "ali@example.com, cy@example.com", None),
            ("Transpose this CSV table so that each original column becomes a row, keeping the header labels as "
             "the first column:\n\nmetric,Q1,Q2,Q3\nrevenue,120,135,150\ncosts,80,85,90",
             "metric,revenue,costs\nQ1,120,80\nQ2,135,85\nQ3,150,90", None),
        ],
    },
}


def choose_validation_families(rng, groups: dict) -> list[str]:
    """Two families per group, from two different domains (seeded)."""
    out = []
    for group in groups.values():
        domains = list(group)
        for d in [domains[i] for i in sorted(rng.permutation(len(domains))[:2])]:
            fams = list(group[d])
            out.append(fams[int(rng.integers(0, len(fams)))])
    return out


def main():
    aa = {}
    with open(ROOT / "data" / "aa_extraction_questions_all.jsonl") as f:
        for line in f:
            r = json.loads(line)
            aa[int(r["id"])] = r["question"]
    rng = np.random.default_rng(SPLIT_SEED)
    val_fams = choose_validation_families(rng, {"original": ORIGINAL, "checkable": CHECKABLE})
    rows = []
    for domain, fams in ORIGINAL.items():
        for fam, ids in fams.items():
            for qid in ids:
                rows.append({"id": f"orig_aa{qid:03d}", "text": aa[qid],
                             "source": f"assistant-axis@a989619 data/extraction_questions.jsonl id={qid}",
                             "group": "original", "domain": domain, "family": fam,
                             "split": "validation" if fam in val_fams else "discovery",
                             "answer": None, "tests": None})
    for domain, fams in CHECKABLE.items():
        for fam, items in fams.items():
            for i, (text, answer, tests) in enumerate(items, 1):
                rows.append({"id": f"chk_{fam}_{i:02d}", "text": text, "source": f"gshift {VERSION} (authored)",
                             "group": "checkable", "domain": domain, "family": fam,
                             "split": "validation" if fam in val_fams else "discovery",
                             "answer": answer if tests is None else None,
                             "reference_solution": answer if tests is not None else None, "tests": tests})
    ids = [r["id"] for r in rows]
    assert len(rows) == 60 and len(set(ids)) == 60
    for g in ("original", "checkable"):
        n_disc = sum(r["split"] == "discovery" for r in rows if r["group"] == g)
        assert n_disc == 20, (g, n_disc)
    # verify the code tasks' reference solutions pass their own tests
    for r in rows:
        if r["tests"]:
            ns: dict = {}
            exec(r["reference_solution"], ns)
            for t in r["tests"]:
                assert eval(t, ns), (r["id"], t)
    out = ROOT / "data" / "questions" / f"manifest_{VERSION}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and "--force" not in sys.argv:
        old = out.read_text()
        new = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        if old != new:
            sys.exit(f"{out} exists and differs; the manifest is frozen (use --force only before any run)")
        print("manifest unchanged")
        return
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}: validation families = {val_fams}")


if __name__ == "__main__":
    main()
