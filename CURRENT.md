# CURRENT — the live recipe

The golden record: what actually runs on this fleet today, pinned. **Serving is sparkrun**
([`sparkrun/README.md`](sparkrun/README.md)). CI enforces two things against this file: the launcher
hashes at the bottom of it against `scripts/launch/`, so a launcher change without a `CURRENT.md`
change fails the build, and recipe↔launcher argv parity via
`scripts/check/test-prompt-token-details.py`, so the primary path cannot drift from the legacy one.
Everything else about *why* lives in [`docs/`](docs/); this file is the *what*.

**Model:** `DeepSeek-V4-Flash-Vision-Exp` @ `86f746b36186f0e567729a5c06a8c918caba82a9` —
the only supported model. **Runtime:** vLLM
`0.21.1rc1.dev339+g1967a5627bc3`, DSpark `k=5` probabilistic, `nvfp4_ds_mla` KV,
1M context. **Served id:** `deepseek-v4-flash-dspark` on `:8888`.

Both launchers read `MODEL_DIR` (default `DeepSeek-V4-Flash-Vision-Exp`); the uncensored build
is the same launcher with `MODEL_DIR=keys-DeepSeekV4Flash-Vision-EXP-ablit` — same shards, same
tokenizer.

## TP2 — asusi + bluey

**Recipe:** [`sparkrun/ds4-vision-exp-tp2_v1.yaml`](sparkrun/ds4-vision-exp-tp2_v1.yaml) →
`sparkrun run ./sparkrun/ds4-vision-exp-tp2_v1.yaml`

**Legacy launcher:** [`scripts/launch/ds4-vision-tp2.sh <0|1>`](scripts/launch/ds4-vision-tp2.sh)
— see [`docs/LEGACY-LAUNCHERS.md`](docs/LEGACY-LAUNCHERS.md) for staging, rank-by-rank launch and
exit codes.

| rank | node | fabric IP | role |
|---|---|---|---|
| 0 | **Asusi** | `192.168.192.3` | head — serves `:8888`, reads weights over NFS from Bluey (`/mnt/bluey-models`) |
| 1 | **Bluey** | `192.168.192.1` | worker (`--headless`), weights local at `/var/tmp/models`, NFS-exports them |

Under sparkrun the ranks are assigned and launched for you; the table is the topology it targets.
The legacy path is worker-first: rank 1, then rank 0. `--master-port 25440`. Plane A only: plane B
(`roceP2p1s0f0`) is not on a common subnet between these two nodes, so `MERGE_NICS` would try to
bring RC QPs up across mismatched subnets.

**Expected (TP2):** single-stream real-prompt decode **≈ 53 tok/s** · KV pool **≈ 2.79M
tokens** · `--max-num-seqs 12` at `--gpu-memory-utilization 0.85`.

## TP4 — all four Sparks

**Recipe:** [`sparkrun/ds4-vision-exp-tp4_v1.yaml`](sparkrun/ds4-vision-exp-tp4_v1.yaml) →
`sparkrun run ./sparkrun/ds4-vision-exp-tp4_v1.yaml`

**Legacy launcher:** [`scripts/launch/ds4-vision-tp4.sh <0|1|2|3>`](scripts/launch/ds4-vision-tp4.sh)
— see [`docs/LEGACY-LAUNCHERS.md`](docs/LEGACY-LAUNCHERS.md).

| rank | node | fabric IP | role |
|---|---|---|---|
| 0 | **Asusi** | `192.168.192.3` | head — serves `:8888` |
| 1 | **Bluey** | `192.168.192.1` | worker (local weights) |
| 2 | **Reddie** | `192.168.192.2` | worker (NFS) |
| 3 | **Spark4** | `192.168.192.4` | worker (NFS) |

Legacy launch order is **3, 2, 1, then 0**, each command run on its own node.

**Recipe deltas from TP2:** same image, same `k=5` probabilistic DSpark, same
`nvfp4_ds_mla` KV, same `--gpu-memory-utilization 0.85` and `--max-model-len 1048576`;
`--tensor-parallel-size 4`, `--nnodes 4`, **`--max-num-seqs 64`** and
**`--max-cudagraph-capture-size 66`** (`= 11×(1+k)`, the smallest capture bucket covering 64
requests — see [`docs/LAUNCH-FLAGS.md`](docs/LAUNCH-FLAGS.md)). CUDA graphs are on
(`--enforce-eager` is not passed and the head log shows `Graph capturing finished in 10 secs,
took 0.68 GiB`); the launcher sets no cudagraph-mode env var, so the mode is the image's own
default rather than something this recipe pins.

**Expected (TP4):**

| | |
|---|---|
| KV cache pool | **8,328,795 tokens** |
| Time to healthy | **~7 min** |
| Real-prompt decode, single stream | **prose 42 tok/s · code 98 tok/s** |
| Mixed, 16 streams | **124 tok/s aggregate** |
| Counting ceiling (draft-acceptance only) | **95 tok/s at C1 · 1,073 tok/s at C48** |
| Cold prefill | **~4.6K tok/s**, flat from 14K to 182K tokens |

## Preflight

Under sparkrun the payload is staged inside the container at boot and verified before the server
starts, so there is nothing to pre-stage by hand — verify the boot instead (§6 of
[`docs/SPARKRUN-PARITY.md`](docs/SPARKRUN-PARITY.md)). What each staged file is and which patch it
carries: [`docs/PATCHES.md`](docs/PATCHES.md).

- Patch 6 keeps the prefix cache alive on long conversations: both tiers pass
  `VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION=${PROTECTED_FRACTION:-0.30}` and
  `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS=${SWA_RECYCLE:-1}`. Measured on TP2: a 354K-token prompt
  re-sent after another 354K of prefill hits 100% in 1.2 s where the stock image gives 0% and a
  235 s cold re-prefill.
- Both tiers pass `--enable-prompt-tokens-details`, so clients can read
  `usage.prompt_tokens_details.cached_tokens`
  ([`docs/CACHE-REPORTING.md`](docs/CACHE-REPORTING.md)).
- **Drop the page cache on all nodes before launch.**
- Before quoting any number, confirm the `B12X` MoE backend line in the boot log: a missing Patch 4
  mount costs about half your decode speed with perfect output.
- The legacy launcher path stages by hand on every node and fails closed on missing files
  (`exit 4`); see [`docs/LEGACY-LAUNCHERS.md`](docs/LEGACY-LAUNCHERS.md).

## Reading the numbers above

Decode is quoted from **real prompts** (prose, code); counting-prompt figures are labeled
draft-acceptance ceilings and are never throughput; prefill is **cold only**. The full
discipline — warm-up and idle decay, streaming under-reports under spec decode, KV pool being a
per-boot figure, reporting conditions with the number — lives in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Repo conventions

- **`sparkrun/` is the serving path.** `ds4-vision-exp-tp2_v1.yaml` and
  `ds4-vision-exp-tp4_v1.yaml` are self-contained — each rebuilds the runtime in-container from a
  pinned source commit ([`sparkrun/README.md`](sparkrun/README.md)) — and serve the same
  `deepseek-v4-flash-dspark` id on `:8888` as the launchers. Every deliberate difference
  between a recipe and its launcher is enumerated in
  [`docs/SPARKRUN-PARITY.md`](docs/SPARKRUN-PARITY.md), and
  `scripts/check/test-prompt-token-details.py` fails the build on undocumented drift.
- **The shell launchers in `scripts/launch/` are the legacy path.** They still work and are kept
  for the unpublished local fleet image, for air-gapped hosts with weights already staged, and as
  the argv baseline the parity check runs against. The tables above describe the serving profile
  both paths reach. Operating instructions:
  [`docs/LEGACY-LAUNCHERS.md`](docs/LEGACY-LAUNCHERS.md).
- **`.env.dspark` is the fleet env template**
  ([`.env.dspark.example`](.env.dspark.example)) for the scripts that source it —
  `scripts/build/build-dspark-vllm-runtime.sh`, `scripts/serve/prepare-dspark-model-cache.sh`
  and `scripts/serve/smoke-deepseek-v4-flash-dspark.sh`. The launchers take their
  configuration from their own `-e` blocks plus `PROTECTED_FRACTION` / `SWA_RECYCLE` /
  `MODEL_DIR` from the ambient shell, and do **not** source `.env.dspark`.
- **Serving is k=5 probabilistic DSpark at `--max-model-len 1048576`**, thinking **on** by
  default (`--default-chat-template-kwargs '{"thinking":true}'`). Reasoning comes back on the
  `reasoning` field (there is no `reasoning_content` on this runtime) and it consumes
  `max_tokens` before any content is produced — cap requests at 32K, not 8K.
- **Vision-Exp is the only supported model.**

<!-- launcher hashes, maintained by scripts/check/check-current.sh --write -->
sha256 ee15082c434154c844096bd5aeb0ade37fe5f424f97eb6c214db76580e09f128  scripts/launch/ds4-vision-tp2.sh
sha256 56d4664aceab18d1ed701863a9f8c5f671aa5ed6845f880bebf69ca4e3b51fdd  scripts/launch/ds4-vision-tp4.sh
