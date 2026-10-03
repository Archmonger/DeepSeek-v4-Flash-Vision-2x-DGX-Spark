# DeepSeek-V4-Flash-Vision-Exp on DGX Spark

Reproducible recipe for serving **[`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)** on NVIDIA DGX Spark with
vLLM, DSpark speculative decoding, and the `nvfp4_ds_mla` NVFP4 KV cache:
**1M-token context with native image input**, and concurrency that stays clean under
agent traffic. TP2 (2 nodes) and TP4 (4 nodes).

> **[`CURRENT.md`](CURRENT.md) is the source of truth** — pinned commits, node/rank maps,
> preflight, and the numbers we actually measure. This page is the entry point; the
> reference material lives in [`docs/`](docs/).

This repo is **Vision-Exp only**.

## Quick start

Stage the bind-mounted files on **every** node first — the launcher checks for them but
does not copy them for you.

```bash
# 1. Generate the four vision-port files (run on every node)
./scripts/build/build-ds4v-files.sh      # -> /var/tmp/ds4v_{model,vision,mm,registry}.py

# 2. Stage the patched runtime files (run on every node)
#    What each file is and why it is required: docs/PATCHES.md
cp recipe/overlay/vllm/v1/core/sched/scheduler.py                  /var/tmp/patch3-scheduler.py
cp recipe/overlay/vllm/v1/spec_decode/dspark.py                    /var/tmp/spec-dspark.py
cp recipe/overlay/vllm/v1/core/single_type_kv_cache_manager.py     /var/tmp/patch6-single_type_kv_cache_manager.py

# 3. Launch worker-first — TP2: rank 1 (worker), then rank 0 (head)
./scripts/launch/ds4-vision-tp2.sh 1
./scripts/launch/ds4-vision-tp2.sh 0

# 4. Verify the API and the patches
curl -fsS http://127.0.0.1:8888/v1/models
./scripts/check/check-patch4.sh <head-container> <worker-container>
```

**TP4:** `./scripts/launch/ds4-vision-tp4.sh <rank>`, launched in rank order
**3 → 2 → 1 → 0**. Cluster addresses, fabric IPs, and the rank map are in
[`CURRENT.md`](CURRENT.md).

The API serves at `http://<head>:8888/v1` under the model id
**`deepseek-v4-flash-dspark`**.

### Verify before you trust any number

The launcher fails closed on missing staged files, but one class of failure is silent: a
missing patch mount can cost roughly **half your decode speed while producing perfect
output and no error**. Run the fail-closed preflight on **both** nodes before quoting any
measurement, and see [`docs/PATCHES.md`](docs/PATCHES.md) for the full patch inventory,
what each patch does, how each is delivered, and how to verify it landed.

```bash
./scripts/check/check-patch4.sh <head-container> <worker-container>   # run against BOTH nodes
```

## Patches

Everything about the patches — inventory, symptoms, root causes, delivery (baked into the
image vs read-only bind mount), staging, verification, and the environment knobs — lives in
one place: **[`docs/PATCHES.md`](docs/PATCHES.md)**.

## What the recipe pins

| | |
|---|---|
| **Checkpoint** | `DeepSeek-V4-Flash-Vision-Exp` @ `86f746b36186f0e567729a5c06a8c918caba82a9` |
| **Drop-in variant** | same launcher with `MODEL_DIR=keys-DeepSeekV4Flash-Vision-EXP-ablit` |
| **Topology** | TP2 (`--nnodes 2`) or TP4 (`--nnodes 4`), `--distributed-executor-backend mp` |
| **Context** | `--max-model-len 1048576` |
| **KV cache** | `--kv-cache-dtype nvfp4_ds_mla`, `--block-size 256` (Stage C padded envelope — see [troubleshooting](docs/TROUBLESHOOTING.md#the-nvfp4-path-is-the-stage-c-padded-envelope)) |
| **Concurrency** | `--max-num-seqs 12` (TP2) · `64` (TP4) |
| **Speculative decoding** | DSpark, `num_speculative_tokens: 5`, `draft_sample_method: probabilistic` |
| **Serving** | port `8888` · `--enable-prefix-caching` · `--enable-prompt-tokens-details` · `--async-scheduling` · `--enable-chunked-prefill` |
| **Reasoning** | off by default (`--default-chat-template-kwargs '{"thinking":false}'`) — see [reasoning mode](docs/REASONING-MODE.md) |

KV pool size is a **per-boot** figure that swings with unified-memory usage; read it off
the boot log of the boot you are quoting ([`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

## Where to read what

| I want to… | Go to |
|---|---|
| know exactly what runs today, pinned | [`CURRENT.md`](CURRENT.md) |
| understand the TP2 command flag by flag | [`docs/LAUNCH-FLAGS.md`](docs/LAUNCH-FLAGS.md) |
| understand the patches | [`docs/PATCHES.md`](docs/PATCHES.md) |
| work with thinking / `reasoning_effort` | [`docs/REASONING-MODE.md`](docs/REASONING-MODE.md) |
| read cache hits per request | [`docs/CACHE-REPORTING.md`](docs/CACHE-REPORTING.md) |
| debug a broken or slow deployment | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| benchmark without fooling myself | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| find the script that does X | [`scripts/README.md`](scripts/README.md) |
| understand the vision port internals | [`vision-exp/README.md`](vision-exp/README.md) |
| run this on hardware that isn't the author's | [`docs/PORTABILITY.md`](docs/PORTABILITY.md) |
| one-command deployment via sparkrun | [`sparkrun/README.md`](sparkrun/README.md) |
| evaluate a move to stock vLLM official main / v0.24+ | [`docs/UPGRADE-OFFICIAL-MAIN.md`](docs/UPGRADE-OFFICIAL-MAIN.md) |
| see who did what | [`CREDITS.md`](CREDITS.md) |

## Repository layout

| path | purpose |
|---|---|
| `scripts/` | **every runnable script**, grouped by job — see [`scripts/README.md`](scripts/README.md) |
| `scripts/launch/` | the two runnable launchers: `ds4-vision-tp2.sh <0\|1>`, `ds4-vision-tp4.sh <0\|1\|2\|3>` |
| `scripts/build/` | image build (`build-dspark-vllm-runtime.sh`), vision-port file generation, overlay source check |
| `scripts/serve/` | model-cache prep (`prepare-dspark-model-cache.sh`) and the end-to-end smoke test (`smoke-deepseek-v4-flash-dspark.sh`) |
| `scripts/check/` | the CI guard (`check-current.sh`) and the fail-closed preflights (`check-patch3.sh`, `check-patch4.sh`) |
| `scripts/bench/` | measurement harnesses (peak, soak, concurrency, garble taps) |
| `scripts/diagnose/`, `scripts/patching/`, `scripts/experimental/` | output-shape analysis, in-place patchers, alternate-runtime lanes |
| `recipe/` | runtime overlay sources, the overlay Dockerfile, and the NVFP4 stage A/B/C Dockerfiles |
| `patches/` | patch files and patchers applied to the runtime |
| `vision-exp/` | the vision port payload — `port/*.py` (patchers + `ds4v_*` sources), consumed by `scripts/build/build-ds4v-files.sh` |
| `docs/` | reference docs — patches, launch flags, reasoning mode, cache reporting, troubleshooting, benchmarking, portability, the upgrade path |
| `sparkrun/` | self-contained sparkrun recipes |

## Contributing

If you change a launcher or any serving flag:

1. Update `CURRENT.md` in the same PR, and say which line of it your change moves.
2. Run `bash scripts/check/check-current.sh --write` (CI runs `bash scripts/check/check-current.sh`).
3. Quote performance from **real prompts**, warm, with the token count attached — see
   [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## License

Repo scripts and docs are under [`LICENSE`](LICENSE) (MIT). The vLLM overlay and runtime
files and the patches under `patches/` are vLLM/DSpark-derived and retain their
Apache-2.0 lineage and SPDX headers where present. Base images,
FlashInfer/TileLang/Triton/CUDA/NCCL, and model weights are separate upstream
artifacts with their own licenses and terms. Attribution: [`CREDITS.md`](CREDITS.md).
