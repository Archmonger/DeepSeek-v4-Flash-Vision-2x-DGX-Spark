# Agent Garble Fix - Fresh C12 NVFP4 DSpark Profile

This note is for anyone who already cloned or deployed this repo and then saw
agent traffic degrade into repeated characters, Chinese drift, leaked tool/XML
prompt text, or unstable loops.

## ⚠️ FIRST: is Patch 3 actually loaded?

**If you still get garble after applying everything else in this document, this is why.**

Patch 3 (credit **@roady001**, issue #3) is the **cold-start garble root-cause fix**. It lives in
the scheduler, not in any sampling flag, so none of the settings below can substitute for it.

**The canonical image `vllm-dspark-runtime:dspark-nvfp4-stage-c` has it baked in.** But the
older `probe-c-p2b` image **predates Patch 3** and needs it bind-mounted. A pre-Patch-3 image
boots clean, passes smoke tests, and serves correctly on warm requests — it only garbles on
**cold prefill**, which is why it survives every quick check and then bites in production.

### Check it in 5 seconds

```bash
docker exec <container> grep -c is_prefill_chunk \
  /opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py
```

- `5` → Patch 3 is loaded. Good.
- `0` → **Patch 3 is MISSING.** Fix it before changing anything else.

### Fix

```bash
# on every node
cp recipe/overlay/vllm/v1/core/sched/scheduler.py /var/tmp/patch3-scheduler.py

# add to the container run
-v /var/tmp/patch3-scheduler.py:/opt/env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:ro
```

### What it fixes, and what it looks like when it's missing

Patch 3 guards the spec-placeholder resize so it only runs for requests the AsyncScheduler
would actually give placeholders. Without the guards the resize runs for **every** running
request — including ones mid **chunked prefill** — attaching spec tokens to the final prompt
chunk of a long **cold** resume and corrupting the prompt tail.

Symptoms, all of which we measured on a 2x DGX Spark with a ~20k-token agent prompt:

- reply opens with **prompt echo** or **leaked tool/skill-schema text**
- reply **starts mid-word** (`'s, and tools.'`, `'ThatGaming: Skills for...'`)
- sometimes leading special tokens (`<|begin_of_sentence|>#`)
- **recovers as soon as the request is warm** — prefix-cache hits never fail

### Measured, 2026-07-30 (2x DGX Spark, ~18k-token system prompt + 58 tools)

Forced cold prefill on every request, real captured agent payload:

| config | cold-prefill failures |
| --- | ---: |
| k=5, as-shipped | 11/12 |
| k=5 + `VLLM_DSPARK_CONFIDENCE_THRESHOLD=0.1` | 12/12 |
| **k=3** | **10/10** |
| k=5 + every other setting in this document | 10/10 |
| **k=5 + Patch 3** | **0/10** ✅ |
| **k=3 + Patch 3** | **0/10** ✅ |

Warm requests: **0/19 failures in every configuration.**

**Two conclusions worth internalising:**

1. **`k` is not the variable.** Dropping `num_speculative_tokens` from 5 to 3 is a widely-repeated
   "fix" for this garble. It does not work — k=3 failed 10/10 on cold prefill. If it appeared to
   help you historically, you almost certainly had Patch 3 loaded and k=3 got the credit. Use
   **k=5** (with Patch 3) and keep the ~24% decode that k=3 costs you.
2. **A clean 5-prompt gate proves nothing here.** Warm requests never fail. You must force a cold
   prefill to see it — see the reproducer below.

### Reproducer

[`benchmarks/replay_hermes.py`](benchmarks/replay_hermes.py) replays a captured agent request and
forces a cold prefill each iteration by prepending a unique nonce to the system prompt (busting
the prefix cache), then scores the output for prompt echo, schema dumps, mid-word starts and
special-token leakage.

```bash
# capture real agent traffic first (transparent proxy, records prompt+response)
UPSTREAM=http://<lane>:8888 PORT=8890 python3 benchmarks/garble_tap.py
#   ... point your agent at :8890 for a while ...

# then replay it cold against any candidate config
URL=http://<lane>:8888/v1 N=10 COLD=1 python3 benchmarks/replay_hermes.py
```

Patch 3 also made cold prefills **faster** in our runs (~36s → ~12s), because spec tokens are no
longer being wrongly attached to prefill chunks.

---

The fix is not to drop DeepSeek V4 Flash DSpark, switch to a smaller fallback,
or move production to fp8. The current stable path keeps:

- `kv_cache_dtype=nvfp4_ds_mla`
- `max_model_len=1500000`
- `max_num_seqs=12`
- `MTP_NUM_TOKENS=3` with `draft_sample_method=probabilistic`
- Keys Patch 2b concurrency behavior
- no server-side sampling override (`--generation-config vllm` only)

> **2026-07-03 update.** The primary garble — a new session's first prompt
> dumping tool-call fragments under concurrency, then recovering — was
> root-caused to a DSpark spec-decode cold-start draft/target mismatch (a greedy
> draft) plus a `repetition_penalty` crash risk, not sampling. The current fix is
> the changes summarized in
> [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) and
> [`docs/PATCHES.md`](docs/PATCHES.md). This note is kept for the
> deployment-drift and per-node checks that still apply.
>
> **2026-07-31 correction.** The "greedy draft" half of that root cause does not
> hold: `draft_sample_method` is a **no-op** for DSpark in this runtime — the
> proposer only populates draft probabilities under
> `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1`, so greedy and probabilistic take the same
> rejection-sampler path. The actual root cause is Patch 3 (spec placeholders
> installed on chunked-prefill requests); see
> [`docs/PATCHES.md`](docs/PATCHES.md). Setting `draft_sample_method` is harmless
> but changes nothing.

## What Was Happening

The bad symptom usually appeared only under real agent traffic. Basic direct
prompts like `hi` could look fine, while Hermes/OpenClaw-style long prompts
with tools, schemas, and concurrent sessions could drift or loop.

The failures we isolated came from a mix of deployment drift and unsafe defaults:

1. A reused runtime image tag can hide an older or partial DSpark overlay.
2. Two-node launches can bind the worker to the head node's fabric IP unless
   `VLLM_HOST_IP` is explicit per node.
3. Agent clients can inherit unstable model-card sampling unless the server
   overrides generation defaults.
4. Harness testing can be contaminated by stale sessions or silent fallbacks.
5. Some worker nodes need their own checkout path and Hugging Face cache path.

As of the 2026-07-03 garble fix there is no server-side sampling override at all
(the launcher runs `--generation-config vllm` with no `--override-generation-config`).
Client request parameters remain the source of truth; do not add a server-side
`repetition_penalty`, which is a documented DSpark spec-decode crash risk.

## What Changed

The public recipe now carries the stable agent-serving defaults directly:

- `--speculative-config '{"method":"dspark","num_speculative_tokens":3,"draft_sample_method":"probabilistic"}'`
- `--max-cudagraph-capture-size` equal to `--max-num-seqs`
- `--async-scheduling` and `--enable-chunked-prefill`
- `--default-chat-template-kwargs '{"thinking":false}'`
- `--generation-config vllm` (no `--override-generation-config`; the old
  `repetition_penalty=1.05` was a spec-decode crash risk)
- `VLLM_USE_FLASHINFER_SAMPLER=1`
- `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`
- `VLLM_DSPARK_CONFIDENCE_SCHEDULER=off`
- `VLLM_DSPARK_LOCAL_ARGMAX=1`
- `VLLM_DSPARK_REPLICATE_MARKOV_W1=1`
- `VLLM_DSPARK_FUSED_MARKOV_ARGMAX=0`
- `VLLM_DSV4_B12X_COMPRESSED_MLA=0`
- `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`
- explicit `VLLM_HOST_IP` and `WORKER_VLLM_HOST_IP`
- explicit `WORKER_SCRIPT_DIR` and `WORKER_HF_CACHE`

## Update Path For Existing Deployments

From the head node checkout:

```bash
git pull
cp .env.dspark.example .env.dspark.new
```

Copy your node-specific values from the old `.env.dspark` into the new file.
At minimum verify these values:

```bash
WORKER_HOST=...
WORKER_SCRIPT_DIR=...
MASTER_ADDR=...
NCCL_IB_HCA=...
NCCL_SOCKET_IFNAME=...
NCCL_IB_GID_INDEX=...
HF_CACHE=...
WORKER_HF_CACHE=...
VLLM_HOST=0.0.0.0
VLLM_HOST_IP=...
WORKER_VLLM_HOST_IP=...
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=12
MAX_NUM_BATCHED_TOKENS=8192
GPU_MEMORY_UTILIZATION=0.85
MTP_NUM_TOKENS=3
VLLM_USE_FLASHINFER_SAMPLER=1
VLLM_DSPARK_REPLICATE_MARKOV_W1=1
```

Then replace the old env file:

```bash
mv .env.dspark .env.dspark.before-garble-fix
mv .env.dspark.new .env.dspark
```

Rebuild both node images so stale local tags cannot keep the old overlay alive:

```bash
./build-dspark-vllm-runtime.sh
```

Restart worker-first:

```bash
./stop-deepseek-v4-flash-dspark.sh
./start-deepseek-v4-flash-dspark.sh
```

## Verify Before Pointing Agents At It

Direct API first:

```bash
curl -fsS http://HEAD_NODE_IP:8888/v1/models
```

Confirm the model reports:

```json
"max_model_len": 1500000
```

Then run a deterministic chat check:

```bash
curl -fsS http://HEAD_NODE_IP:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash-dspark",
    "messages": [{"role": "user", "content": "Reply exactly: NVFP4 DSPARK OK"}],
    "max_tokens": 8,
    "temperature": 0
  }'
```

Expected text:

```text
NVFP4 DSPARK OK
```

Check logs:

```bash
docker compose --env-file .env.dspark -f docker-compose.dspark.yml logs vllm-dspark \
  | grep -E "GPU KV cache size|Maximum concurrency|Application startup complete|generation_config|override_generation_config"
```

Expected shape:

```text
GPU KV cache size: about 3.2M tokens
Maximum concurrency for 1,500,000 tokens per request: about 2.1x
Application startup complete.
```

If you need the prior conservative agent lane, use:

```bash
MAX_MODEL_LEN=1048576
MAX_NUM_SEQS=6
GPU_MEMORY_UTILIZATION=0.80
```

Do not enable `VLLM_USE_B12X_FP8_GEMM=1` on the Stage C image. It selected an
experimental dense FP8 path but failed DSpark drafter warmup during validation.

## Agent Harness Rules

Only after direct vLLM prompts are clean, point Hermes/OpenClaw/other agents to:

```text
http://HEAD_NODE_IP:8888/v1
model: deepseek-v4-flash-dspark
context_length: 1048576
temperature: 0
thinking: false
```

Do not set a `repetition_penalty` on the DSpark speculative-decode path; it is a
documented spec-decode crash risk (illegal memory access), not a garble fix.

For the prior conservative 1M/6 lane, use `context_length: 1048576`.

During validation:

- disable hidden fallbacks to Qwen/27B/other models
- clear or restart stale sessions if a session already garbled
- test one direct prompt, then 2/4/6 concurrent prompts, then agent traffic
- use explicit `temperature: 0` only for exact deterministic curl checks

If direct vLLM is clean but agent traffic still garbles, the remaining problem
is probably harness/session/fallback state, not the DSpark weights.
