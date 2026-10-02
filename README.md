# DeepSeek-V4-Flash-Vision-Exp on DGX Spark

Reproducible recipe for serving **[`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)** on NVIDIA DGX Spark with
vLLM, DSpark speculative decoding, and the `nvfp4_ds_mla` NVFP4 KV cache:
**1M-token context with native image input**, and concurrency that stays clean under
agent traffic. TP2 (2 nodes) and TP4 (4 nodes).

> **[`CURRENT.md`](CURRENT.md) is the source of truth** — pinned commits, node/rank maps,
> preflight, and the numbers we actually measure. This page is the entry point; the
> reference material lives in [`docs/`](docs/).

> DeepSeek-V4.1-Flash is a different model with its own recipe:
> [tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-V4.1-Flash-vLLM-DGX-Spark).
> This repo is Vision-Exp only.

## Quick start

Stage the bind-mounted files on **every** node first — the launcher checks for them but
does not copy them for you.

```bash
# 1. Generate the four vision-port files (run on every node)
./vision-exp/build-ds4v-files.sh          # -> /var/tmp/ds4v_{model,vision,mm,registry}.py

# 2. Stage the three patched runtime files (run on every node)
cp recipe/overlay/vllm/v1/core/sched/scheduler.py                  /var/tmp/patch3-scheduler.py
cp recipe/overlay/vllm/v1/spec_decode/dspark.py                    /var/tmp/spec-dspark.py
cp recipe/overlay/vllm/v1/core/single_type_kv_cache_manager.py     /var/tmp/patch6-single_type_kv_cache_manager.py

# 3. Launch worker-first — TP2: rank 1 (worker), then rank 0 (head)
./launchers/ds4-vision-tp2.sh 1
./launchers/ds4-vision-tp2.sh 0

# 4. Verify the API and the patches
curl -fsS http://127.0.0.1:8888/v1/models
./scripts/check-patch4.sh <head-container> <worker-container>
```

**TP4:** `./launchers/ds4-vision-tp4.sh <rank>`, launched in rank order
**3 → 2 → 1 → 0**. Cluster addresses, fabric IPs, and the rank map are in
[`CURRENT.md`](CURRENT.md).

The API serves at `http://<head>:8888/v1` under the model id
**`deepseek-v4-flash-dspark`**.

### Check Patch 4 before you trust any number

A missing `spec-dspark.py` mount loads the DSpark draft's always-on shared expert
**uninitialised**. Result: roughly **half the decode speed with perfect output quality
and no error** — the dropped tensors are reported at `logger.debug`, invisible at the
default log level, and the broken load reports success. That combination sends you
looking in exactly the wrong place.

```bash
./scripts/check-patch4.sh <head-container> <worker-container>   # run against BOTH nodes
```

Full mechanism: [`DSPARK-SHARED-EXPERT-FIX.md`](DSPARK-SHARED-EXPERT-FIX.md).

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
| **Reasoning** | off by default (`--default-chat-template-kwargs '{"thinking":false}'`) — see [reasoning mode](docs/reasoning-mode.md) |

KV pool size is a **per-boot** figure that swings with unified-memory usage; read it off
the boot log of the boot you are quoting ([`docs/BENCHMARKS.md`](docs/BENCHMARKS.md)).

## Patches

All four are required. The launcher hard-fails if Patch 3, 4, or 6 is missing from
`/var/tmp`; a missing vision-port file fails at container start instead.

| Patch | What it fixes | Source in this repo | Staged at | Reference |
|---|---|---|---|---|
| **3** | Cold-start agent garble (spec-placeholder resize guard) | `recipe/overlay/vllm/v1/core/sched/scheduler.py` | `/var/tmp/patch3-scheduler.py` | [`docs/PATCHES.md`](docs/PATCHES.md) |
| **4** | DSpark draft shared-expert loads uninitialised → silent half speed | `recipe/overlay/vllm/v1/spec_decode/dspark.py` | `/var/tmp/spec-dspark.py` | [`DSPARK-SHARED-EXPERT-FIX.md`](DSPARK-SHARED-EXPERT-FIX.md) |
| **6** | Prefix cache lost on long conversations | `recipe/overlay/vllm/v1/core/single_type_kv_cache_manager.py` | `/var/tmp/patch6-single_type_kv_cache_manager.py` | [`docs/PATCH6-KV-CACHE-PREFIX-EVICTION.md`](docs/PATCH6-KV-CACHE-PREFIX-EVICTION.md) |
| **Vision port** | Native image input (ViT + aligner + multimodal registry alias) | `vision-exp/port/*.py`, generated per image by `build-ds4v-files.sh` | `/var/tmp/ds4v_{model,vision,mm,registry}.py` | [`vision-exp/README.md`](vision-exp/README.md) |

Patch 6 is tunable: `PROTECTED_FRACTION` (default `0.30`) and `SWA_RECYCLE`
(default `1`) — see [`CURRENT.md`](CURRENT.md).

The overlay image (`recipe/Dockerfile.dspark-runtime-overlay`) bakes the same overlay
sources in at build time; the vision launchers use read-only bind mounts because the
deployed image predates the baked patches. Both routes deliver the same code — and a
proposer mounted from a mismatched vLLM build will crash the rig, so match the image.

## Where to read what

| I want to… | Go to |
|---|---|
| know exactly what runs today, pinned | [`CURRENT.md`](CURRENT.md) |
| understand the TP2 command flag by flag | [`VISION-EXP-DEFAULT-CONFIG.md`](VISION-EXP-DEFAULT-CONFIG.md) |
| understand the patches (1 / 2 / 2b / 3 / 5 / 6) | [`docs/PATCHES.md`](docs/PATCHES.md) |
| work with thinking / `reasoning_effort` | [`docs/reasoning-mode.md`](docs/reasoning-mode.md) |
| read cache hits per request | [`docs/cache-reporting.md`](docs/cache-reporting.md) |
| debug a broken or slow deployment | [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) |
| benchmark without fooling myself | [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) |
| understand the vision port internals | [`vision-exp/README.md`](vision-exp/README.md) |
| run this on hardware that isn't the author's | [`docs/PORTABILITY.md`](docs/PORTABILITY.md) |
| one-command deployment via sparkrun | [`sparkrun/README.md`](sparkrun/README.md) |
| compare our serving fidelity to the hosted reference | [`parity/`](parity/) |
| see who did what | [`CREDITS.md`](CREDITS.md) |

## Repository layout

| path | purpose |
|---|---|
| `launchers/` | the two runnable launchers: `ds4-vision-tp2.sh <0\|1>`, `ds4-vision-tp4.sh <0\|1\|2\|3>` |
| `vision-exp/` | the vision port — `port/*.py` and `build-ds4v-files.sh`, which stages the four bind-mounted files per image |
| `recipe/` | runtime overlay sources, the overlay Dockerfile, and the NVFP4 stage A/B/C Dockerfiles |
| `patches/` | patch files and patchers applied to the runtime |
| `scripts/` | preflight checks (`check-patch3.sh`, `check-patch4.sh`), sanity benches, guards |
| `benchmarks/` | measurement harnesses and captured checkpoint evidence |
| `docs/` | reference docs — patches, reasoning mode, troubleshooting, benchmarking, portability, cache reporting |
| `sparkrun/` | self-contained sparkrun recipes |
| `parity/` | reproducible serving-fidelity bench + frozen hosted reference card |
| `tools/` | repo maintenance; `check-current.sh` keeps `CURRENT.md`'s launcher hashes honest |

## Contributing

If you change a launcher or any serving flag:

1. Update `CURRENT.md` in the same PR, and say which line of it your change moves.
2. Run `bash tools/check-current.sh --write` (CI runs `bash tools/check-current.sh`).
3. Quote performance from **real prompts**, warm, with the token count attached — see
   [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## License

Repo scripts and docs are under [`LICENSE`](LICENSE) (MIT). The vLLM overlay and runtime
files and the patches under `patches/` are vLLM/DSpark-derived and retain their
Apache-2.0 lineage and SPDX headers where present. Base images,
FlashInfer/TileLang/Triton/CUDA/NCCL, and model weights are separate upstream
artifacts with their own licenses and terms. Attribution: [`CREDITS.md`](CREDITS.md).
