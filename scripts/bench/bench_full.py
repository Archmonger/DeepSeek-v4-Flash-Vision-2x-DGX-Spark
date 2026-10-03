#!/usr/bin/env python3
"""Full DS4 characterization: peak decode, concurrency scaling, prefill.

Three axes, because single-stream decode alone doesn't describe a serving box:
  1. peak/mean decode by content type (acceptance-driven)
  2. concurrency scaling c1..c6 (aggregate + per-stream)
  3. prefill throughput at depth (TTFT for big prompts)

Env: URL, MODEL, TAG
"""
import json
import os
import threading
import time
import urllib.request

URL = os.environ.get("URL", "http://192.168.192.2:8889")
MODEL = os.environ.get("MODEL", "deepseek-v4-flash-dspark")
TAG = os.environ.get("TAG", "current")


def post(prompt, max_tokens, temp=0.0, timeout=600):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temp}
    req = urllib.request.Request(URL + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    u = r["usage"]
    return u["completion_tokens"], u.get("prompt_tokens", 0), dt


PEAK = [
    ("count300", "Print the numbers 1 to 300, one per line, exact format N. No commentary.", 1200),
    ("mult12",   "Print the full 12x12 multiplication table, one line per pair, format A x B = C. No commentary.", 900),
    ("json60",   'Output a JSON array of 60 objects, each exactly {"id":N,"name":"user_N","active":true}. JSON only.', 800),
    ("bst",      "Implement a binary search tree in Python with insert, search, delete, in-order traversal, "
                 "docstrings and usage examples. Code only.", 600),
    ("story",    "Write a 200-word story about an engineer debugging a distributed system at 3am.", 400),
]

CONC_PROMPT = ("Implement a binary search tree in Python with insert, search, delete and in-order "
               "traversal. Include docstrings and two usage examples. Code only.")


# Heavy warm-up is MANDATORY on a freshly booted engine, not politeness.
# Measured 2026-07-29: immediately after "Application startup complete" (graphs already
# captured), count300 ran at 58.5 tok/s; after ~5 long generations it settled at 83.3.
# A 30% penalty, invisible in the boot log. Short 100-token calls do NOT clear it --
# it takes several hundred-token generations. Benchmark cold and you measure the cold path.
WARMUP = [
    ("Implement a binary search tree in Python with insert, search, delete and in-order "
     "traversal. Include docstrings and two usage examples.", 700),
    ("Write a 300-word explanation of how speculative decoding works.", 600),
    ("Compute the running sum of the first 40 prime numbers, showing each step.", 600),
    ('Output a JSON array of 40 objects, each {"id":N,"name":"user_N"}. JSON only.', 600),
    ("Explain tensor parallelism versus pipeline parallelism in detail.", 600),
]


def warm(short=4):
    """Drive the engine to steady state: long generations first, then short calls."""
    for prompt, mt in WARMUP:
        try:
            post(prompt, mt)
        except Exception:
            pass
    for _ in range(short):
        try:
            post("Write a python function that adds two numbers. Code only.", 100)
        except Exception:
            pass


def bench_peak():
    print(f"\n--- [{TAG}] DECODE by content (temp 0, warm) ---")
    print(f"{'prompt':<10}{'tok':>6}{'sec':>7}{'tok/s':>9}")
    vals = []
    for label, p, mt in PEAK:
        try:
            a = post(p, mt); b = post(p, mt)
            ct, _, dt = a if a[0]/a[2] > b[0]/b[2] else b
            tps = ct / dt
            vals.append((label, tps))
            print(f"{label:<10}{ct:>6}{dt:>7.2f}{tps:>9.1f}", flush=True)
        except Exception as e:
            print(f"{label:<10}  FAILED {str(e)[:40]}")
    if vals:
        v = [x[1] for x in vals]
        top = max(vals, key=lambda x: x[1])
        print(f"  PEAK {top[1]:.1f} ({top[0]})   MEAN {sum(v)/len(v):.1f}")
        return top[1], sum(v)/len(v)
    return 0, 0


def bench_conc():
    print(f"\n--- [{TAG}] CONCURRENCY (same prompt, 400 tok each) ---")
    print(f"{'conc':>5}{'ok':>5}{'agg tok/s':>12}{'per-stream':>12}{'wall':>8}")
    out = {}
    for c in (1, 2, 4, 6):
        results = []
        lock = threading.Lock()

        def worker():
            try:
                ct, _, dt = post(CONC_PROMPT, 400)
                with lock:
                    results.append((ct, dt))
            except Exception:
                pass

        ts = [threading.Thread(target=worker) for _ in range(c)]
        t0 = time.time()
        for t in ts: t.start()
        for t in ts: t.join()
        wall = time.time() - t0
        if results:
            tot = sum(r[0] for r in results)
            agg = tot / wall
            per = sum(r[0] / r[1] for r in results) / len(results)
            out[c] = agg
            print(f"{c:>5}{len(results):>5}{agg:>12.1f}{per:>12.1f}{wall:>8.2f}", flush=True)
        else:
            print(f"{c:>5}    0        FAILED")
    return out


def bench_prefill():
    print(f"\n--- [{TAG}] PREFILL (TTFT method, 1 output token) ---")
    print(f"{'target':>8}{'prompt tok':>12}{'sec':>8}{'tok/s':>10}")
    filler = ("Distributed inference on GB10 schedules prefill and decode in the same step; "
              "long prompts dominate the step budget and delay in-flight decodes. ")
    res = {}
    for target in (8000, 32000, 100000):
        try:
            n = max(1, int(target / 22))
            prompt = (filler * n)[:target * 4] + "\nSummarize in one sentence."
            ct, pt, dt = post(prompt, 1, timeout=900)
            res[target] = pt / dt
            print(f"{target:>8}{pt:>12}{dt:>8.1f}{pt/dt:>10.0f}", flush=True)
        except Exception as e:
            print(f"{target:>8}   FAILED {str(e)[:40]}")
    return res


if __name__ == "__main__":
    print(f"=== {TAG} @ {URL} ===")
    warm()
    pk, mn = bench_peak()
    cc = bench_conc()
    pf = bench_prefill()
    print(f"\n===== SUMMARY [{TAG}] =====")
    print(f"decode peak {pk:.1f} | mean {mn:.1f}")
    if cc:
        print("concurrency agg: " + "  ".join(f"c{k}={v:.0f}" for k, v in cc.items()))
    if pf:
        print("prefill: " + "  ".join(f"{k//1000}K={v:.0f}t/s" for k, v in pf.items()))
