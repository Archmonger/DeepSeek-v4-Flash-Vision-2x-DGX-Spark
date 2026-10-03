# Credits

This repo combines several public efforts. Please credit the upstream authors
when reusing the recipe, the patch, or benchmark numbers.

## DSpark Concurrency Patch

The in-server DSpark concurrency breakthrough comes from Keys / drowzeys:

- Repo: https://github.com/drowzeys/Keys-Concurrency-Patch-for-DSpark-DeepSeek-V4-Flash
- Tested commit in this repo: `7e4d94bbcec95223550517c0fa9244e59f9f6483`

Keys' patch fixes the two core blockers for `max_num_seqs > 1`:

- Request-stable DSpark main-KV slots, so persistent DSpark draft KV follows
  request identity instead of condensed vLLM batch-row position.
- Ragged `query_start_loc` handling for real independent-arrival batches where
  prefill and decode rows mix in the same scheduler step.

The validated concurrency numbers in this repo depend directly on that patch.

## DSpark Cold-Start Garble Root-Cause Fix (Patch 3)

The scheduler-level root cause of the cold-resume garble (prompt echo / leaked tool-schema text
at the start of a reply on long resumed conversations) was tracked down and fixed as a
collaboration between **Roady001** and **Fable**:

- **Roady001** — reported the issue, showed that the launch/config workarounds in circulation
  only reduce the symptom, and independently validated the final fix on his own 2x DGX Spark.
  Issue: https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/issues/3
- **Fable** — the root-cause analysis and the patch.
  Fix commit: https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/commit/e83606a

Mechanism, the `Scheduler.update_from_output` guard and how to verify it landed:
[`docs/PATCHES.md`](docs/PATCHES.md) (Patch 3).

## CUDA-Graph Capture-Size Fix (concurrency throughput)

**Wpnx330** found and fixed a silent throughput cliff: a `--max-cudagraph-capture-size` that
floors below the workload drops concurrency off the captured CUDA-graph path into
eager/piecewise. What the flag does in this recipe:
[`docs/LAUNCH-FLAGS.md`](docs/LAUNCH-FLAGS.md).

- PR: https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/5

## DSpark vLLM Integration

Rafael Caricio published the DSpark vLLM integration and deployment work this
recipe builds on:

- https://github.com/rafaelcaricio/vllm/pull/1
- https://github.com/rafaelcaricio/spark_vllm_docker/pull/1

## DSpark Draft-Loading Work

Fraser Price published the DSpark-on-vLLM drafter integration and draft-weight loading
work that this recipe's runtime builds on:

- https://github.com/fraserprice/dspark-vllm

## Two-Node DGX Spark Packaging

MiaAI-Lab published the two-node DGX Spark packaging and launch lineage this
repo builds from:

- https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark

## Upstream Foundations

This work also relies on:

- vLLM
- FlashInfer
- NVIDIA CUDA/NCCL/Blackwell tooling
- `DeepSeek-V4-Flash-Vision-Exp` and the DeepSeek V4 architecture
- DeepSeek-AI DeepSpec / DSpark speculative decoding research

## TonyD2Wild Contribution

This repo contributes the validated 2x DGX Spark NVFP4-KV recipe, Stage A/B/C
runtime packaging, sanitized two-node launch flow, application of Keys'
concurrency patch to the NVFP4 profile, and benchmark artifacts from the
validated runs.

## License Notes

Repo-local scripts and docs are MIT licensed via `LICENSE`.

The vLLM overlay files and `patches/keys-concurrency.patch` are vLLM/DSpark
derived and retain their Apache-2.0 lineage from the upstream sources and
Keys' patch repo. Model weights, base images, CUDA/NCCL, FlashInfer, TileLang,
and Triton are separate upstream artifacts with their own licenses and terms.

## 0rand

- Parameterized the API port (`VLLM_PORT`, PR #1).
- Independently identified speculation-linked output garbling and proposed a shallower draft depth in PR #1. The durable fix is the scheduler-level guard in Patch 3, and the validated draft depth here is `k=5` (see [`docs/PATCHES.md`](docs/PATCHES.md)).

## paulbrav

- Reported the long-context engine-death crash (#2) with a clean deterministic repro AND shipped the fix (PR #4: sparse-indexer gather guard + stale draft-KV slot clamp).
- Producer-side instrumentation of the slot-corruption mechanism (canary readback, 12h compute-sanitizer, graph-replay-private-state localization) and a debug-tooling dead-ends writeup (cuda-gdb/coredump limits on GB10/sbsa) that materially informs #6.

## DaveCharland

- Reported and characterized the episodic soft-failure (#6): real output tokens parsing to empty/thinking-only content under sustained agent load, with a deterministic within-episode repro and full spec-acceptance-collapse telemetry.
