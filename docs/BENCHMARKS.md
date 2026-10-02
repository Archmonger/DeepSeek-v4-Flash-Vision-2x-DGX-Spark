# How to benchmark (and how to read a number)

Measured throughput for the current recipe lives in
[`CURRENT.md`](../CURRENT.md). This page is the discipline that makes those numbers
comparable — most bogus "the model is slower/faster" reports come from breaking one of
the rules below.

## Quote decode from real prompts, never from the counting prompt

`Count from 1 to 300` is nearly perfectly predictable, so the DSpark drafter accepts
almost every token it proposes. That figure measures **how fast speculation can run**,
not how fast the model serves work. `CURRENT.md` labels counting figures as
draft-acceptance ceilings and keeps them out of the throughput line.

## Decode = steps/s × accepted-tokens-per-step

Acceptance is content-driven, so any single-prompt "tok/s" for this model is a
statement about the prompt as much as the hardware. The same server legitimately
spans roughly 30 → 80 tok/s across content types. Quote the **mean** for planning,
the **peak** for bragging, and always name the content and the token count.

## Warm up — and the warm state decays when idle

The first requests after `Application startup complete` run **~30% slow**, even with
CUDA graphs already captured and a few short warm-up calls sent. The server reports
itself ready, answers correctly, and is simply slow for a while.

- A handful of 100-token calls is **not** enough. Steady state took gate-sized
  (500–700 token) generations to reach.
- Once reached it is stable: consecutive runs agreed within 0.2 tok/s.
- **Idling re-introduces the penalty.** After roughly 30 idle minutes the same prompt
  measured cold again and needed heavy warm-up to recover. This hits a fleet after any
  quiet period, not just after a deploy — so never benchmark straight after a lull.

Most of the spread in casually reported numbers for this stack is this effect.

## Short requests are mathematically capped

Fixed per-request overhead is roughly 0.5 s, which dominates short generations:

```
134 tokens / 80 tok/s + 0.5 s overhead = 2.18 s  ->  61 tok/s apparent
measured:                                2.27 s  ->  59.1 tok/s
```

That is overhead amortisation, **not** poor acceptance. A short request cannot post a
high tok/s no matter how predictable its content. Use long generations
(1,200–1,400 tokens) when you are looking for the ceiling.

## Under speculative decoding, streaming under-reports

vLLM emits at most one SSE chunk per decode **step**, carrying every token accepted
in that step. Counting streamed content deltas therefore measures **steps/s, not
tokens/s**, and under-reports by the accepted-length factor. For throughput tests use
`"stream": false`, read `usage.completion_tokens`, or divide the server's
`vllm:generation_tokens_total` by wall time.

Reserve streaming tests for streaming behaviour — latency-to-first-token, `delta`
shape, `stream_options.include_usage` — not for tok/s.

## Prefill: cold only

Prefix caching is on, so a "warm" prefill number measures the cache, not the model.
Report cold prefill explicitly, and report the prompt depth with it.

## Aggregate vs per-stream

Concurrency multiplies aggregate throughput and divides per-stream throughput. Both
are real; they answer different questions. Plan capacity from a **mixed-traffic**
measurement at your target concurrency, not from a single-prompt benchmark scaled up —
templated benchmark prompts are much more predictable than real agent traffic.

## KV pool size is a per-boot figure

Two boots of an identical config can differ by ~11% in reported KV pool: available KV
memory on GB10 depends on what else has touched unified memory. Read the pool off the
boot log of the boot you are quoting, and treat headroom against the 1M context ceiling
as the thing that matters.

## Harnesses in this repo

| script | what it measures |
| --- | --- |
| [`benchmarks/bench_full.py`](../benchmarks/bench_full.py) | decode by content type, concurrency sweep, prefill depth |
| [`benchmarks/soak.py`](../benchmarks/soak.py) | long runs of realistic mixed agent traffic (throughput + stability) |
| [`benchmarks/realwork_peak.py`](../benchmarks/realwork_peak.py) | which real output shapes approach the ceiling |
| [`benchmarks/garble_tap.py`](../benchmarks/garble_tap.py) | concurrent-output corruption tap |
| [`scripts/agent_sanity_bench.py`](../scripts/agent_sanity_bench.py) | 1/2/4/6-concurrency sanity + garble check before pointing a harness at the endpoint |

If the direct bench is clean but an agent harness reports garbage, investigate the
harness, its fallback list, and prompt replay before blaming the model — see
[Troubleshooting](TROUBLESHOOTING.md).

## Report the conditions with the number

A number is not comparable without: temperature, warm vs cold, prompt content **and
token count**, concurrency, patch set mounted, image tag and vLLM build string. When
you open an issue, include those; when you quote someone else's number, check for them
first.
