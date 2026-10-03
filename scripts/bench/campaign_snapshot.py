#!/usr/bin/env python3
"""Bluey performance snapshot. Built against the three traps that burned us:
  - SSE chunk counting (under-reports ~4x under spec decode): non-streaming only,
    usage.completion_tokens over wall clock.
  - prefix-cache hits (repeated prompts inflate up to 9x): every request carries a
    unique nonce at the FRONT of the prompt.
  - content-dependent acceptance (count=97%, prose=40%): five prompt classes,
    reported separately, never pooled into one number.
Decode tok/s uses the trap-80 method: a max_tokens=1 prefill floor per prompt,
then (completion_tokens-1)/(total_s - floor_s).
Usage: snapshot.py <base_url> <model> <label> <out.json>"""
import json, sys, time, uuid, re, urllib.request, concurrent.futures as cf, statistics as st

BASE, MODEL, LABEL, OUT = sys.argv[1:5]
BASE = BASE.rstrip("/")
CHAT = (BASE if BASE.endswith("/v1") else BASE + "/v1") + "/chat/completions"
METRICS = (BASE[:-3] if BASE.endswith("/v1") else BASE) + "/metrics"

def post(messages, max_tokens, temperature=0.0, timeout=900):
    body = json.dumps({"model": MODEL, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temperature, "stream": False}).encode()
    t0 = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(CHAT, body, {"Content-Type": "application/json"}), timeout=timeout))
    return r, time.time() - t0

def nonce(): return f"[ref {uuid.uuid4().hex[:10]}] "

def metrics():
    try:
        t = urllib.request.urlopen(METRICS, timeout=20).read().decode()
    except Exception:
        return None
    g = lambda n: sum(float(m.group(1)) for m in re.finditer(rf"^{n}\S*\s+([0-9.e+]+)$", t, re.M))
    return {"accepted": g("vllm:spec_decode_num_accepted_tokens_total"), "drafted": g("vllm:spec_decode_num_draft_tokens_total")}

# ---------- 1. decode battery ----------
CLASSES = {
    "chat":  ("Give me three practical tips for sleeping better, one sentence each.", 200),
    "count": ("Count from 1 to 300, comma separated, nothing else.", 400),
    "code":  ("Write a Python function that parses an ISO-8601 date string and returns a datetime, with docstring and three example usages. Only code.", 400),
    "prose": ("Explain in about 250 words why the sky is blue, for a curious teenager.", 400),
    "tool":  ('Extract to JSON with keys who, when, where, duration_min: "Lunch with Priya next Thursday at noon at Ember Kitchen, should take about an hour and a half." Return only the JSON.', 120),
}
REPS = 5
res = {"label": LABEL, "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "decode": {}, "prefill": {}, "concurrency": {}}
m0 = metrics()
for name, (prompt, mt) in CLASSES.items():
    rows = []
    for i in range(REPS):
        p = nonce() + prompt
        _, floor = post([{"role": "user", "content": p}], 1)          # prefill floor, same prompt
        r, total = post([{"role": "user", "content": p}], mt)
        n = r["usage"]["completion_tokens"]
        dec = (n - 1) / max(total - floor, 1e-3)
        rows.append({"tokens": n, "total_s": round(total, 3), "floor_s": round(floor, 3), "decode_tok_s": round(dec, 2),
                     "finish": r["choices"][0]["finish_reason"]})
        print(f"  decode/{name:5} rep{i+1}: {n:4d} tok  {dec:7.2f} tok/s", flush=True)
    ds = [x["decode_tok_s"] for x in rows]
    res["decode"][name] = {"rows": rows, "mean": round(st.mean(ds), 2), "median": round(st.median(ds), 2), "peak": round(max(ds), 2)}
m1 = metrics()
if m0 and m1 and m1["drafted"] > m0["drafted"]:
    res["acceptance_pct"] = round(100 * (m1["accepted"] - m0["accepted"]) / (m1["drafted"] - m0["drafted"]), 2)
    print(f"  acceptance over battery: {res['acceptance_pct']}%")
allmeans = [v["mean"] for v in res["decode"].values()]
res["decode_battery_mean"] = round(st.mean(allmeans), 2)

# ---------- 2. prefill / TTFT ----------
WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu".split()
def synth(target_tokens):
    import random
    rnd = random.Random(uuid.uuid4().int)
    return nonce() + " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 999)) for _ in range(int(target_tokens * 0.55)))
for tgt in (1000, 8000, 32000, 128000):
    rows = []
    for i in range(2):
        p = synth(tgt)
        r, t = post([{"role": "user", "content": p + "\n\nReply with the single word OK."}], 1, timeout=1800)
        pt = r["usage"]["prompt_tokens"]
        rows.append({"prompt_tokens": pt, "ttft_s": round(t, 3), "prefill_tok_s": round(pt / t, 1)})
        print(f"  prefill {tgt:>6}: {pt:6d} tok in {t:7.2f}s = {pt/t:8.1f} tok/s", flush=True)
    res["prefill"][str(tgt)] = {"rows": rows, "ttft_s_mean": round(st.mean(x["ttft_s"] for x in rows), 3),
                                "prefill_tok_s_mean": round(st.mean(x["prefill_tok_s"] for x in rows), 1)}

# ---------- 3. concurrency ----------
def one(prompt, mt):
    r, t = post([{"role": "user", "content": nonce() + prompt}], mt)
    return r["usage"]["completion_tokens"], t
for c in (1, 2, 4):
    jobs = [(CLASSES["count"][0], 400), (CLASSES["prose"][0], 400)] * 2
    jobs = jobs[:c] if c < 4 else jobs
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        outs = list(ex.map(lambda j: one(*j), jobs))
    wall = time.time() - t0
    tot = sum(n for n, _ in outs)
    res["concurrency"][f"c{c}"] = {"streams": c, "total_tokens": tot, "wall_s": round(wall, 2),
                                   "aggregate_tok_s": round(tot / wall, 2), "per_stream_tok_s": round(tot / wall / c, 2)}
    print(f"  c{c}: {tot} tok in {wall:.1f}s = {tot/wall:.1f} aggregate, {tot/wall/c:.1f}/stream", flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print(f"\nBATTERY MEAN {res['decode_battery_mean']} tok/s  |  written {OUT}")
