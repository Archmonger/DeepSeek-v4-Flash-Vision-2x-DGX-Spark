#!/usr/bin/env python3
"""Which REAL-WORK prompt shapes reach the acceptance ceiling (~84 tok/s)?

count300 hits 84.3 but it's a toy. The question is whether any genuinely useful output shape
gets there. Hypothesis from the per-position acceptance data: acceptance is ~100% when the next
5 tokens are near-deterministic given context. That should hold for:
  - rigid repeated templates (fixtures, DTOs, config, SQL inserts)
  - transformations where the output is largely determined by the input
  - boilerplate with a fixed skeleton
and fail for anything requiring novel word choice.

Env: URL, MODEL
"""
import json
import os
import time
import urllib.request

URL = os.environ.get("URL", "http://192.168.192.2:8889/v1")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")

TABLE = """id,region,units,unit_price
1,north,12,4.50
2,south,7,9.25
3,east,19,3.10
4,west,4,15.00
5,north,22,2.75
6,south,15,6.40
7,east,9,11.20
8,west,31,1.95
9,north,6,18.30
10,south,27,3.85"""

CASES = [
    # --- the toy baseline, for reference ---
    ("count300 (toy)",
     "Print the numbers 1 to 300, one per line, exact format N. No commentary.", 1200),

    # --- rigid repeated templates: bulk data / fixtures ---
    ("sql-inserts",
     "Generate 60 SQL INSERT statements for table users(id, email, created_at). "
     "Use the exact form: INSERT INTO users (id, email, created_at) VALUES (N, "
     "'user_N@example.com', '2026-01-01'); — ids 1 to 60, one per line. SQL only.", 1200),

    ("json-fixtures",
     'Output a JSON array of 60 objects, each EXACTLY {"id":N,"sku":"SKU-N",'
     '"qty":N,"active":true} with N from 1 to 60. JSON only, no commentary.', 1200),

    ("dataclasses",
     "Write 20 Python dataclasses named Item1 through Item20. Each has exactly the fields "
     "id: int, name: str, value: float and a method total(self) -> float that returns "
     "self.value * self.id. Identical structure each time. Code only.", 1200),

    ("env-config",
     "Write a .env file with 60 entries in the exact form SERVICE_N_TIMEOUT_MS=1000 "
     "for N from 1 to 60. No commentary.", 1000),

    # --- transformation: output largely determined by input ---
    ("csv-to-json",
     "Convert this CSV to a JSON array of objects, preserving field names and order, "
     "one object per row. JSON only.\n\n" + TABLE, 900),

    ("csv-to-md",
     "Convert this CSV into a GitHub-flavoured markdown table, same columns, same order, "
     "no extra commentary.\n\n" + TABLE, 700),

    ("add-types",
     "Add type hints to every function. Return the complete file, unchanged otherwise, "
     "code only.\n\n"
     "def add(a, b):\n    return a + b\n\n"
     "def scale(xs, k):\n    return [x * k for x in xs]\n\n"
     "def total(rows):\n    return sum(r['units'] * r['price'] for r in rows)\n\n"
     "def label(name, count):\n    return f'{name}: {count}'\n\n"
     "def clamp(v, lo, hi):\n    return max(lo, min(hi, v))", 700),

    # --- boilerplate code with a fixed skeleton ---
    ("crud-endpoints",
     "Write FastAPI CRUD endpoints for 8 resources: user, order, product, invoice, "
     "shipment, refund, coupon, review. For each, exactly four routes (GET list, GET by id, "
     "POST create, DELETE) with identical structure and a TODO body. Code only.", 1400),

    # --- control: genuinely novel prose ---
    ("prose (control)",
     "Write 200 words of original prose about an engineer debugging a distributed system.", 400),
]


def post(prompt, max_tokens, temp=0.0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    req = urllib.request.Request(URL + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=600))
    dt = time.time() - t0
    return r["usage"]["completion_tokens"], dt


if __name__ == "__main__":
    # HEAVY warm-up, mandatory. The warm state decays with idle time, not just at boot:
    # ~30 min idle after a soak, count300 measured 60.4 tok/s with only 2 short warm-up calls,
    # against 83-84 when properly warm. Under-warming here would depress the early cases and
    # silently manufacture a "shape" effect that is really a warm-up gradient.
    print(f"target {URL}   (heavy warm-up first)", flush=True)
    for p, mt in [
        ("Implement a binary search tree in Python with insert, search, delete and in-order "
         "traversal, with docstrings and two usage examples.", 700),
        ("Write a 300-word explanation of how speculative decoding works.", 600),
        ("Compute the running sum of the first 40 primes, showing each step.", 600),
        ("Print the numbers 1 to 200, one per line, format N. No commentary.", 800),
        ("Explain tensor parallelism versus pipeline parallelism in detail.", 600),
    ]:
        try:
            post(p, mt)
        except Exception:
            pass
    for _ in range(3):
        try:
            post("Write a python function that adds two numbers. Code only.", 100)
        except Exception:
            pass
    ct, dt = post("Print the numbers 1 to 300, one per line, exact format N. No commentary.", 1200)
    print(f"warm check: count300 = {ct/dt:.1f} tok/s (expect 83-84; if much lower, still cold)",
          flush=True)
    print(f"\n{'shape':<20}{'tok':>6}{'sec':>7}{'tok/s':>9}")
    print("-" * 42)
    rows = []
    for label, prompt, mt in CASES:
        try:
            a = post(prompt, mt)
            b = post(prompt, mt)
            ct, dt = a if a[0] / a[1] > b[0] / b[1] else b
            tps = ct / dt
            rows.append((label, tps))
            print(f"{label:<20}{ct:>6}{dt:>7.2f}{tps:>9.1f}", flush=True)
        except Exception as e:
            print(f"{label:<20}  FAILED {str(e)[:34]}", flush=True)
    print("-" * 42)
    for label, tps in sorted(rows, key=lambda r: -r[1]):
        print(f"  {tps:6.1f}  {label}")
