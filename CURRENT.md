# CURRENT — the live recipe

The golden record: what actually runs on this fleet today, pinned. CI checks the launcher
hashes at the bottom of this file against `launchers/`, so a launcher change without a
`CURRENT.md` change fails the build. Everything else about *why* lives in
[`docs/`](docs/); this file is the *what*.

**Model:** `DeepSeek-V4-Flash-Vision-Exp` @ `86f746b36186f0e567729a5c06a8c918caba82a9` —
the only supported model. **Runtime:** vLLM
`0.21.1rc1.dev339+g1967a5627bc3`, DSpark `k=5` probabilistic, `nvfp4_ds_mla` KV,
1M context. **Served id:** `deepseek-v4-flash-dspark` on `:8888`.

Both launchers read `MODEL_DIR` (default `DeepSeek-V4-Flash-Vision-Exp`); the uncensored build
is the same launcher with `MODEL_DIR=keys-DeepSeekV4Flash-Vision-EXP-ablit` — same shards, same
tokenizer.

## TP2 — asusi + bluey

**Launcher:** [`launchers/ds4-vision-tp2.sh <0|1>`](launchers/ds4-vision-tp2.sh)

| rank | node | fabric IP | role |
|---|---|---|---|
| 0 | **Asusi** | `192.168.192.3` | head — serves `:8888`, reads weights over NFS from Bluey (`/mnt/bluey-models`) |
| 1 | **Bluey** | `192.168.192.1` | worker (`--headless`), weights local at `/var/tmp/models`, NFS-exports them |

**Launch order: 1 (worker), then 0 (head).** `--master-port 25440`. Plane A only: plane B
(`roceP2p1s0f0`) is not on a common subnet between these two nodes, so `MERGE_NICS` would try
to bring RC QPs up across mismatched subnets.

**Expected (TP2):** single-stream real-prompt decode **≈ 53 tok/s** · KV pool **≈ 2.79M
tokens** · `--max-num-seqs 12` at `--gpu-memory-utilization 0.85`.

## TP4 — all four Sparks

**Launcher:** [`launchers/ds4-vision-tp4.sh <0|1|2|3>`](launchers/ds4-vision-tp4.sh)

| rank | node | fabric IP | role |
|---|---|---|---|
| 0 | **Asusi** | `192.168.192.3` | head — serves `:8888` |
| 1 | **Bluey** | `192.168.192.1` | worker (local weights) |
| 2 | **Reddie** | `192.168.192.2` | worker (NFS) |
| 3 | **Spark4** | `192.168.192.4` | worker (NFS) |

**Launch order: 3, 2, 1, then 0.**

```bash
./launchers/ds4-vision-tp4.sh 3     # spark4
./launchers/ds4-vision-tp4.sh 2     # reddie
./launchers/ds4-vision-tp4.sh 1     # bluey
./launchers/ds4-vision-tp4.sh 0     # asusi (head, serves :8888)
```

**Recipe deltas from TP2:** same image, same `k=5` probabilistic DSpark, same
`nvfp4_ds_mla` KV, same `--gpu-memory-utilization 0.85` and `--max-model-len 1048576`;
`--tensor-parallel-size 4`, `--nnodes 4`, **`--max-num-seqs 64`** and
**`--max-cudagraph-capture-size 64`**. CUDA graphs are on (`--enforce-eager` is not passed and
the head log shows `Graph capturing finished in 10 secs, took 0.68 GiB`); the launcher sets no
cudagraph-mode env var, so the mode is the image's own default rather than something this
recipe pins.

**Expected (TP4):**

| | |
|---|---|
| KV cache pool | **8,328,795 tokens** |
| Time to healthy | **~7 min** |
| Real-prompt decode, single stream | **prose 42 tok/s · code 98 tok/s** |
| Mixed, 16 streams | **124 tok/s aggregate** |
| Counting ceiling (draft-acceptance only) | **95 tok/s at C1 · 1,073 tok/s at C48** |
| Cold prefill | **~4.6K tok/s**, flat from 14K to 182K tokens |

## Preflight, every node

- Stage the patched runtime files at `/var/tmp` **and** the vision-port files. TP2's launcher
  checks `patch3-scheduler.py`, `spec-dspark.py` and `patch6-single_type_kv_cache_manager.py`;
  TP4 additionally checks all four `ds4v_*.py` files. Missing file ⇒ **exit 4**.
  What each file is, where it comes from and how to verify it landed →
  [`docs/PATCHES.md`](docs/PATCHES.md).
- Patch 6 keeps the prefix cache alive on long conversations: both launchers pass
  `VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION=${PROTECTED_FRACTION:-0.30}` and
  `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS=${SWA_RECYCLE:-1}`. Measured on TP2: a 354K-token prompt
  re-sent after another 354K of prefill hits 100% in 1.2 s where the stock image gives 0% and a
  235 s cold re-prefill.
- Both launchers and `docker-compose.dspark.yml` pass `--enable-prompt-tokens-details`, so
  clients can read `usage.prompt_tokens_details.cached_tokens`
  ([`docs/CACHE-REPORTING.md`](docs/CACHE-REPORTING.md)).
- Workers mount Bluey's weights export at `/mnt/bluey-models`.
- **Drop the page cache on all nodes before launch.**
- Before quoting any number, run [`scripts/check-patch4.sh`](scripts/check-patch4.sh) against
  **both** nodes: a missing Patch 4 mount costs about half your decode speed with perfect output.

## Reading the numbers above

Decode is quoted from **real prompts** (prose, code); counting-prompt figures are labeled
draft-acceptance ceilings and are never throughput; prefill is **cold only**. The full
discipline — warm-up and idle decay, streaming under-reports under spec decode, KV pool being a
per-boot figure, reporting conditions with the number — lives in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## Repo conventions

- **The vision launchers in `launchers/` are the supported path**, and they are what the tables
  above describe.
- **`sparkrun/` carries a self-contained Vision-Exp recipe** that rebuilds the same runtime
  in-container ([`sparkrun/README.md`](sparkrun/README.md)). It serves under the id
  `deepseek-v4-flash-vision-exp`; the launchers serve `deepseek-v4-flash-dspark`. Clients
  pointed at `:8888` use the launcher's id.
- **The Compose files are the generic two-node serve/build configuration**:
  `docker-compose.dspark.yml` plus [`.env.dspark.example`](.env.dspark.example) — the template
  for the `.env.dspark` that `build-dspark-vllm-runtime.sh` also sources.
- **Serving is k=5 probabilistic DSpark at `--max-model-len 1048576`**, thinking off by
  default ([`docs/REASONING-MODE.md`](docs/REASONING-MODE.md)).
- **Vision-Exp is the only supported model.**

<!-- launcher hashes, maintained by tools/check-current.sh --write -->
sha256 a2deb28b31a0ce105cf21703ab031a3d731122e8ea5dcf69a2bf5302cce6cc38  launchers/ds4-vision-tp2.sh
sha256 484e1b2eee241989c93c4dda5d3e1e4370a237e631be25b2d0a4946de1abd2b3  launchers/ds4-vision-tp4.sh
