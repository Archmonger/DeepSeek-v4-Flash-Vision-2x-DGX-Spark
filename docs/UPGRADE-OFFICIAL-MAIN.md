# Upgrading to stock vLLM official main / v0.24+ with DSpark

What upstream merges give this recipe, and what they do not. Live pins: [`../CURRENT.md`](../CURRENT.md) · patch detail: [`PATCHES.md`](PATCHES.md) · measurement discipline: [`BENCHMARKS.md`](BENCHMARKS.md) · other-hardware gotchas: [`PORTABILITY.md`](PORTABILITY.md).

## Bottom line

- DSpark PR `vllm-project/vllm#46995` is merged and is the right lane for the garble/concurrency bug class.
- Stock official vLLM main / v0.24 does **not** boot this NVFP4 DSpark deployment on 2× DGX Spark (SM120/GB10). Do not replace the current runtime with stock.
- The recipe stays **k=5 probabilistic DSpark** at `--max-model-len 1048576`, `--max-num-seqs 12` on TP2 and `64` on TP4 — pinned identically in the primary sparkrun recipes ([`../sparkrun/ds4-vision-exp-tp2_v1.yaml`](../sparkrun/ds4-vision-exp-tp2_v1.yaml), [`../sparkrun/ds4-vision-exp-tp4_v1.yaml`](../sparkrun/ds4-vision-exp-tp4_v1.yaml)) and the legacy launchers ([`../scripts/launch/ds4-vision-tp2.sh`](../scripts/launch/ds4-vision-tp2.sh), [`../scripts/launch/ds4-vision-tp4.sh`](../scripts/launch/ds4-vision-tp4.sh)).
- Rollback posture: keep the current known-good image parked; build any candidate under a new explicit tag.

## Stock official-main boot failures

Image tested: `vllm-dspark-runtime:official-main-dspark-00eb7ce` — vLLM head `00eb7ce`, containing DSpark merge commit `f5a8d73377d0f0a4e00cba172f9fbd0d50471b07` (PR `#46995`). The DSpark modules import cleanly; every MoE backend fails before generation.

| Backend attempt | Result |
|---|---|
| auto / DeepGEMM MXFP4 | `Unknown SF transformation` on SM120 |
| DeepGEMM disabled / Marlin fallback | unsupported PTX/toolchain in the Marlin FP4 repack |
| `--moe-backend flashinfer_trtllm` | backend rejects the current CUDA device |
| `--moe-backend flashinfer_cutlass` | selects, then fails MXFP4 method compatibility |
| `--moe-backend triton` | backend rejects the current CUDA device |

## Why this is not a one-flag fix

The mismatch is the MXFP4/NVFP4 oracle wiring: DeepSeek V4 reaches MoE through the MXFP4 path while official main's B12X kernel is hooked to the NVFP4 oracle.

| Layer | official main | this runtime |
|---|---|---|
| B12X kernel | `vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py` — asserts NVFP4, supports `kNvfp4Static` | — |
| Oracle wiring | `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` — NVFP4 only | — |
| DSv4 weight path | `vllm/model_executor/layers/quantization/mxfp4.py` + `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py` → selects `kMxfp4Static` | same MXFP4 selection |
| MXFP4 → B12X bridge | absent | `vllm/model_executor/layers/fused_moe/b12x_moe.py` |
| SM120 model glue | absent | `vllm/models/deepseek_v4/nvidia/sm120.py` |
| Sparse MLA kernel | absent | `vllm/v1/attention/backends/mla/b12x_mla_sparse.py` |
| KV dtype | no `nvfp4_ds_mla` | `nvfp4_ds_mla` wired through config, KV cache, and DeepSeek attention |

Merged under PR `#46995`: the DSpark implementation plus DFlash/DSpark shared-buffer cleanup, non-contiguous Gumbel sampling fix, stale `idx_mapping` fix, padded slot fix, zero dummy-buffer fix, speculative position clamp, and DSpark regression coverage. Do not cherry-pick only the visible garble fixes — they rely on that broader buffer contract.

## The prototype

`vllm-dspark-runtime:official-main-dspark-b12x-nvfp4-proto-00eb7ce` adds a Python-only compatibility overlay on that stock official-main image, carrying the two bridges testing uncovered:

1. DeepSeek V4 `nvfp4` KV requests resolve to `nvfp4_ds_mla`, with `DeepseekV4FlashMLABackend.get_kv_cache_shape(..., "nvfp4_ds_mla")` returning the padded 584-byte layout.
2. Official-main MXFP4 MoE maps `flashinfer_b12x` → `B12xExperts`.

Passed: import smoke, `nvfp4_ds_mla` dtype resolution, B12X MXFP4 oracle selection. Not passed: full model boot, direct generation, 2/4/6 concurrency. Not promoted to any launcher.

Artifacts: [`../patches/official-main-b12x-nvfp4-python.patch`](../patches/official-main-b12x-nvfp4-python.patch) and [`../recipe/official-main/Dockerfile.python-patch`](../recipe/official-main/Dockerfile.python-patch).

## What v0.24.0 helps with

DeepSeek-V4 OOM and memory-planning hardening · MTP projection-prefix naming fixes · supported-KV-cache-dtype fixes · DFlash / FlashInfer / scheduler work across the wider DSv4 family · DeepSeek-V4 attention, prefix-cache, and KV allocation improvements.

Worth staging, still not enough: none of it supplies the SM120 MXFP4 MoE backend, the `nvfp4_ds_mla` KV dtype, or the Keys concurrency behavior this recipe runs on.

## Required port checklist

A compatible image is more than PR `#46995`. It needs:

1. Official DSpark main as the base.
2. An MXFP4 B12X MoE backend integrated into official-main's modular MoE stack.
3. A `flashinfer_b12x` / `b12x_mxfp4` mapping in the MXFP4 oracle.
4. `make_mxfp4_moe_quant_config` support for the B12X W4A16 contract.
5. Weight conversion/retention logic equivalent to the stable runtime's `B12xExperts`.
6. `nvfp4_ds_mla` cache dtype support, or a proven official replacement holding the same 1M-context KV pool and stability.
7. Full direct-API validation before any Hermes/OpenClaw-style agent traffic — staged 262K/fp8 first, then 1M/NVFP4, then 2/4/6 concurrency.

## Validation gate

```bash
./scripts/build/verify-overlay-sources.sh
# launch worker-first on every node, then:
./scripts/check/check-patch4.sh <head-container> <worker-container>
./scripts/serve/smoke-deepseek-v4-flash-dspark.sh
DSPARK_BASE_URL=http://HEAD_NODE_IP:8888/v1 CONCURRENCY=1,2,4,6 \
  python3 scripts/bench/agent_sanity_bench.py
```

Also check: `/v1/models` reports the intended `max_model_len` · no `mtp_block.main_norm` load failure · no BOS / placeholder-token leakage · no CJK drift or repeated-character loops in long direct prompts · no increase in vLLM preemptions under 2/4/6 concurrency · speed measured against the known-good image before switching, per [`BENCHMARKS.md`](BENCHMARKS.md). Deterministic tests send `temperature: 0` in the request body.

## The v0.21 lane as tested then

That lane re-verifies at 1M context on `kv_cache_dtype=nvfp4_ds_mla` with a direct `/v1/chat/completions` request returning `OK`, and is recorded there as `MAX_NUM_SEQS=6` / `MTP_NUM_TOKENS=3` — **the lane as tested then, not the current recipe**, which is k=5 at 12 seqs (TP2) / 64 seqs (TP4) per the launchers above. Its boot reported a GPU KV pool near `2,087,950` tokens, about `1.99x` the 1,048,576-token context per request; every pool and full-context-concurrency figure is a per-boot measurement, not spec — see [`BENCHMARKS.md`](BENCHMARKS.md).
