# The TP2 launch command, flag by flag

The exact `vllm serve` line this recipe runs, and what each flag is doing. The
executable source of truth is [`../scripts/launch/ds4-vision-tp2.sh`](../scripts/launch/ds4-vision-tp2.sh)
(TP2) / [`../scripts/launch/ds4-vision-tp4.sh`](../scripts/launch/ds4-vision-tp4.sh) (TP4);
this page is transcribed from it, and **if the two ever disagree, the launcher wins.**

Verified live: TP=2, **asusi** (rank0/head, serves `:8888`) + **bluey** (rank1/worker),
clean output. Checkpoint `DeepSeek-V4-Flash-Vision-Exp` pinned at
`86f746b36186f0e567729a5c06a8c918caba82a9`; image
`vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b`
(vLLM `0.21.1rc1.dev339+g1967a5627bc3`, B12X MXFP4 MoE). That image predates the baked-in
Patch 3/Patch 4 overlay, so both arrive here as read-only bind mounts.

Not on this page: patch delivery and verification → [`PATCHES.md`](PATCHES.md); node/rank map,
preflight and the live numbers → [`../CURRENT.md`](../CURRENT.md); runtime env (B12X, DSpark,
NCCL/RoCE, JIT-cache split) → [`.env.dspark.example`](../.env.dspark.example) and the launcher
`-e` blocks; how to read a throughput figure → [`BENCHMARKS.md`](BENCHMARKS.md).

```
/opt/env/bin/vllm serve <path-to-DeepSeek-V4-Flash-Vision-Exp> \
  --hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"]}' \
  --served-model-name deepseek-v4-flash-dspark \
  --host 0.0.0.0 --port 8888 \
  --trust-remote-code \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 \
  --kv-cache-dtype nvfp4_ds_mla \
  --block-size 256 \
  --max-model-len 1048576 \
  --max-num-seqs 12 \
  --max-num-batched-tokens 8192 \
  --max-cudagraph-capture-size 12 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --async-scheduling \
  --enable-chunked-prefill \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic"}' \
  --tokenizer-mode deepseek_v4 \
  --distributed-executor-backend mp \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}' \
  --default-chat-template-kwargs '{"thinking":true}' \
  --generation-config vllm \
  --enable-flashinfer-autotune \
  --nnodes 2 --node-rank <0|1> \
  --master-addr <head-fabric-ip> --master-port <port>
```

## Flag notes

- **`--served-model-name deepseek-v4-flash-dspark`** — the established id, deliberately: the
  vision build is a drop-in replacement on the same endpoint, so agents already wired to
  `deepseek-v4-flash-dspark` need no rewiring. The sparkrun recipe serves the same stack under
  `deepseek-v4-flash-vision-exp` instead ([`../sparkrun/README.md`](../sparkrun/README.md)).
- **`--hf-overrides '{"architectures":["DeepseekV4VForConditionalGeneration"]}'`** — selects the
  multimodal registry alias added by `ds4v_registry.py`. Without it vLLM answers
  `is_multimodal_model` from its static arch-name table and rejects images with
  "is not a multimodal model" ([`../vision-exp/README.md`](../vision-exp/README.md)).
- **`--max-model-len 1048576`** — 1M, the checkpoint's true YaRN ceiling
  (`original_max_position_embeddings 65536 × factor 16`), standard on both topologies.
- **`--kv-cache-dtype nvfp4_ds_mla` / `--block-size 256`** — the Stage C padded NVFP4 envelope;
  what that means and what it is not → [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
- **`--gpu-memory-utilization 0.85`** — not a tunable to shave for headroom. 0.80 boots and
  passes smoke tests, then dies under traffic, because DSpark allocates its buffers on the first
  real request.
- **`--max-cudagraph-capture-size`** — must be a **multiple of `1+k`** and **at least
  `--max-num-seqs`**. Under speculative decoding the CUDA-graph capture buckets are multiples
  of `1+k` (`k=5` → 6, 12, 18, …), so a value that floors below your concurrency drops the
  excess requests off the captured path into eager/piecewise and throughput collapses
  **silently** (@Wpnx330,
  [PR #5](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/5)).
  This recipe sets `12` on TP2 (`= 2×6`, covering `--max-num-seqs 12` exactly) and **`66`**
  on TP4 (`= 11×6`, the smallest bucket that covers `--max-num-seqs 64`). Both are asserted
  by `scripts/check/test-prompt-token-details.py`, so a future `--max-num-seqs` bump that
  breaks the rule fails CI rather than costing half of throughput at 3 a.m.
  The drafter-side variant of the same problem is Patch A ([`PATCHES.md`](PATCHES.md)).
- **`--enable-prefix-caching` + `--enable-prompt-tokens-details`** — caching plus the per-request
  `usage.prompt_tokens_details.cached_tokens` readout
  ([`CACHE-REPORTING.md`](CACHE-REPORTING.md)); retention behaviour is Patch 6
  ([`PATCHES.md`](PATCHES.md)).
- **`--speculative-config`** — DSpark, `num_speculative_tokens: 5`, probabilistic draft sampling.
  `k=5` is the validated value for this draft block size; why not `k=3`, and why `k>5` crashes at
  first generation → [`PATCHES.md`](PATCHES.md) (Patch 4, Configuration).
- **`--tokenizer-mode deepseek_v4`** — required for DSpark, and it is what makes
  `chat_template.jinja` inert: prompt formatting comes from the checkpoint's built-in encoder,
  so even an explicit `--chat-template <file>` is accepted, shows up in the engine's
  non-default args, and changes nothing.
- **`--reasoning-parser` / `--reasoning-config` / `--default-chat-template-kwargs`** — thinking
  is **on** by default here (`'{"thinking":true}'`). Reasoning is returned on
  `message.reasoning` (non-streaming) / `delta.reasoning` (streaming); there is **no**
  `reasoning_content` key on this runtime, so clients reading that name see nothing and
  conclude extraction is broken. `<think>` is written into the prompt tail and never
  generated, so a missing opening tag in the completion is correct. Turn it off per request with
  `chat_template_kwargs: {"thinking": false}`. Because reasoning is the default, give requests
  real output budget: a small cap yields `finish_reason: length` with empty `content`.
- **`--generation-config vllm` and nothing else** — no server-side sampling override. A
  `repetition_penalty` on the DSpark path is a crash risk (illegal memory access) and is not a
  garble fix ([`TROUBLESHOOTING.md`](TROUBLESHOOTING.md)).
- **`--async-scheduling`, `--enable-chunked-prefill`, `--enable-flashinfer-autotune`** — part of
  the validated profile; all three belong on this line.
- **`--distributed-executor-backend mp`** with `--nnodes 2` and the `--master-addr`/`--master-port`
  pair; rank 1 adds `--headless`. Ranks, launch order and fabric IPs:
  [`../CURRENT.md`](../CURRENT.md).

Container runtime: `network_mode: host`, `ipc: host`, `shm_size: 64gb`, `gpus: all`,
`-v /dev/infiniband:/dev/infiniband`, memlock unlimited.
