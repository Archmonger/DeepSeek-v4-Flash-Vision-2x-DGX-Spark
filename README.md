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

Serving is **sparkrun**. One command brings up the golden recipe on a cluster; there is no image
build and no private registry.

```bash
uvx sparkrun setup                          # once per cluster: cluster, SSH mesh, ConnectX-7 detection
sparkrun run ./sparkrun/ds4-vision-exp-tp2.yaml     # or ds4-vision-exp-tp4.yaml
```

The recipe pins a public base image by **digest**, fetches this repo at a pinned commit, and
rebuilds the runtime inside the container — overlay, NVFP4 stage A/B/C, Patches 3/4/6 and the
vision port — verifying each step before the server starts. Budget ~200 GB free disk per node and
~9 min cold / ~5 min warm. `Ctrl+C` detaches without killing the job.

| Tier | Recipe |
|---|---|
| TP2 (2 nodes) | [`sparkrun/ds4-vision-exp-tp2.yaml`](sparkrun/ds4-vision-exp-tp2.yaml) |
| TP4 (4 nodes) | [`sparkrun/ds4-vision-exp-tp4.yaml`](sparkrun/ds4-vision-exp-tp4.yaml) |

The API serves at `http://<head>:8888/v1` under the model id **`deepseek-v4-flash-dspark`**.
Full walkthrough, overrides and boot verification:
[`sparkrun/README.md`](sparkrun/README.md) · exact parity against the launchers:
[`docs/SPARKRUN-PARITY.md`](docs/SPARKRUN-PARITY.md).

### Verify before you trust any number

One class of failure is silent: a runtime that came up without Patch 4 costs roughly **half your
decode speed while producing perfect output and no error**. The staging step is designed to abort
rather than serve that way, but confirm it on the boot you are quoting:

```bash
sparkrun logs ds4-vision-exp-tp2 | grep -E "stage-runtime|FATAL"
sparkrun logs ds4-vision-exp-tp2 | grep "Using 'B12X' Mxfp4 MoE backend"
```

A missing B12X line is the half-speed fallback. Full post-boot checklist (every rank): §6 of
[`docs/SPARKRUN-PARITY.md`](docs/SPARKRUN-PARITY.md). The patch inventory itself is
[`docs/PATCHES.md`](docs/PATCHES.md).

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
| **Reasoning** | **on** by default (`--default-chat-template-kwargs '{"thinking":true}'`) · read it from `message.reasoning`, not `reasoning_content` |

KV pool size is a **per-boot** figure that swings with unified-memory usage; read it off
the boot log of the boot you are quoting ([`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

## Where to read what

| I want to… | Go to |
|---|---|
| **run this** | [`sparkrun/README.md`](sparkrun/README.md) |
| know exactly what runs today, pinned | [`CURRENT.md`](CURRENT.md) |
| understand the serve command flag by flag | [`docs/LAUNCH-FLAGS.md`](docs/LAUNCH-FLAGS.md) |
| understand the patches | [`docs/PATCHES.md`](docs/PATCHES.md) |
| read cache hits per request | [`docs/CACHE-REPORTING.md`](docs/CACHE-REPORTING.md) |
| debug a broken or slow deployment | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| benchmark without fooling myself | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| find the script that does X | [`scripts/README.md`](scripts/README.md) |
| understand the vision port internals | [`vision-exp/README.md`](vision-exp/README.md) |
| run this on hardware that isn't the author's | [`docs/PORTABILITY.md`](docs/PORTABILITY.md) |
| check a recipe against the launcher it replaces | [`docs/SPARKRUN-PARITY.md`](docs/SPARKRUN-PARITY.md) |
| run the old way, with the shell launchers | [`docs/LEGACY-LAUNCHERS.md`](docs/LEGACY-LAUNCHERS.md) |
| evaluate a move to stock vLLM official main / v0.24+ | [`docs/UPGRADE-OFFICIAL-MAIN.md`](docs/UPGRADE-OFFICIAL-MAIN.md) |
| see who did what | [`CREDITS.md`](CREDITS.md) |

## Repository layout

| path | purpose |
|---|---|
| `scripts/` | **every runnable script**, grouped by job — see [`scripts/README.md`](scripts/README.md) |
| `scripts/launch/` | **legacy** shell launchers — `ds4-vision-tp2.sh <0\|1>`, `ds4-vision-tp4.sh <0\|1\|2\|3>`; see [`docs/LEGACY-LAUNCHERS.md`](docs/LEGACY-LAUNCHERS.md) |
| `scripts/build/` | image build (`build-dspark-vllm-runtime.sh`), vision-port file generation, overlay source check |
| `scripts/serve/` | model-cache prep (`prepare-dspark-model-cache.sh`) and the end-to-end smoke test (`smoke-deepseek-v4-flash-dspark.sh`) |
| `scripts/check/` | the CI guard (`check-current.sh`) and the fail-closed preflights (`check-patch3.sh`, `check-patch4.sh`) |
| `scripts/bench/` | measurement harnesses (peak, soak, concurrency, garble taps) |
| `scripts/diagnose/`, `scripts/patching/`, `scripts/experimental/` | output-shape analysis, in-place patchers, alternate-runtime lanes |
| `recipe/` | runtime overlay sources, the overlay Dockerfile, and the NVFP4 stage A/B/C Dockerfiles |
| `patches/` | patch files and patchers applied to the runtime |
| `vision-exp/` | the vision port payload — `port/*.py` (patchers + `ds4v_*` sources), consumed by `scripts/build/build-ds4v-files.sh` |
| `sparkrun/` | **the primary serving path** — self-contained recipes per tier |
| `docs/` | reference docs — patches, launch flags, cache reporting, troubleshooting, benchmarking, portability, sparkrun parity, the legacy launchers, the upgrade path |

## Contributing

If you change a serving flag, a recipe or a launcher:

1. Update `CURRENT.md` in the same PR, and say which line of it your change moves.
2. Run `bash scripts/check/check-current.sh --write` (CI runs `bash scripts/check/check-current.sh`).
3. Move the sparkrun recipe and the legacy launcher **together** —
   `python3 scripts/check/test-prompt-token-details.py` fails if one moves without the other, and
   [`docs/SPARKRUN-PARITY.md`](docs/SPARKRUN-PARITY.md) is the list it enforces.
3. Quote performance from **real prompts**, warm, with the token count attached — see
   [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## License

Repo scripts and docs are under [`LICENSE`](LICENSE) (MIT). The vLLM overlay and runtime
files and the patches under `patches/` are vLLM/DSpark-derived and retain their
Apache-2.0 lineage and SPDX headers where present. Base images,
FlashInfer/TileLang/Triton/CUDA/NCCL, and model weights are separate upstream
artifacts with their own licenses and terms. Attribution: [`CREDITS.md`](CREDITS.md).
