Both launchers read `MODEL_DIR` (default `DeepSeek-V4-Flash-Vision-Exp`); the uncensored build is the same launcher with `MODEL_DIR=keys-DeepSeekV4Flash-Vision-EXP-ablit`.

**Expected (TP2):**

- Single-stream **real-prompt decode ≈ 53 tok/s**.
- The counting prompt reaches a higher number; that is the **draft-acceptance ceiling**, not
  throughput. See "How we quote numbers".
- **KV pool ≈ 2.79M tokens.**

---

## DeepSeek-V4-Flash-Vision-Exp, TP4 (all four Sparks)

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

**Preflight, on all four nodes:**

- Patch 3 (`patch3-scheduler.py`), Patch 4 (`spec-dspark.py`), Patch 6
  (`patch6-single_type_kv_cache_manager.py`, source `recipe/overlay/vllm/v1/core/single_type_kv_cache_manager.py`)
  **and** the four vision port files (`ds4v_model.py`, `ds4v_vision.py`, `ds4v_mm.py`, `ds4v_registry.py`)
  staged at `/var/tmp`. The launcher checks all seven and exits if any is missing.
- Patch 6 keeps the prefix cache alive on long conversations: both launchers pass
  `VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION=${PROTECTED_FRACTION:-0.30}` and
  `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS=${SWA_RECYCLE:-1}`. Measured on TP2: a 354K-token prompt
  re-sent after another 354K prefill hits 100% in 1.2 s (stock image: 0%, 235 s cold re-prefill);
  logs in `docs/patch6-validation/`, analysis in `docs/PATCH6-KV-CACHE-PREFIX-EVICTION.md`.
- Both launchers and `docker-compose.dspark.yml` pass `--enable-prompt-tokens-details`,
  so clients can read `usage.prompt_tokens_details.cached_tokens`
  (see [`docs/cache-reporting.md`](docs/cache-reporting.md)).
- Workers mount Bluey's weights export at `/mnt/bluey-models`.
- **Drop page cache on all four nodes before launch.**

**Recipe:** same image, same `k=5` probabilistic DSpark, same `nvfp4_ds_mla` KV, same
`--gpu-memory-utilization 0.85`, `--max-model-len 1048576`. Differences from TP2:
`--tensor-parallel-size 4`, `--nnodes 4`, **`--max-num-seqs 64`**, and
**`--max-cudagraph-capture-size 64`** (the workspace copy of this launcher still said 12/12; the
validated run used 64/64). CUDA graphs are on: `--enforce-eager` is not passed, `--max-cudagraph-capture-size 64` is, and the head log shows `Graph capturing finished in 10 secs, took 0.68 GiB`; the mode is vLLM's default for this image, not pinned by the launcher. Note: the launcher
not set a cudagraph-mode env var, so this is the runtime's own default on this image rather than
something we pin.

**Expected (measured 2026-09-02, TP4):**

| | |
|---|---|
| KV cache pool | **8,328,795 tokens** |
| Time to healthy | **~7 min** |
| Real-prompt decode, single stream | **prose 42 tok/s · code 98 tok/s** |
| Mixed, 16 streams | **124 tok/s aggregate** |
| Counting ceiling (labeled draft-acceptance only) | **95 tok/s at C1 · 1,073 tok/s at C48** |
| Cold prefill | **~4.6K tok/s**, flat from 14K to 182K tokens |

---

## How we quote numbers

- **Decode is quoted from real prompts** — prose, code, and the like. Those are the numbers to
  compare against anything else.
- **The counting prompt is a labeled draft-acceptance ceiling only.** "Count to 300" is nearly
  perfectly predictable, so the DSpark drafter accepts almost every token and the tok/s figure
  measures how fast speculation can run, not how fast the model serves work. Never quote a
  counting number as throughput.
- **Prefill is cold only.** Prefix caching is on, so a warm prefill number measures the cache.

---

## Repo conventions

- **The vision launchers in `launchers/` are the supported path** —
  `launchers/ds4-vision-tp2.sh` and `launchers/ds4-vision-tp4.sh`, which is what the tables above
  describe.
- **`sparkrun/` carries a self-contained Vision-Exp recipe** that rebuilds the same runtime
  in-container ([`sparkrun/README.md`](sparkrun/README.md)). It serves under the id
  `deepseek-v4-flash-vision-exp`; the launchers serve `deepseek-v4-flash-dspark`. Clients pointed
  at `:8888` use the launcher's id.
- **The Compose files are the generic two-node serve/build configuration**:
  `docker-compose.dspark.yml` plus `.env.dspark.example` — the template for the
  `.env.dspark` that `build-dspark-vllm-runtime.sh` also sources.
- **Serving is k=5 probabilistic DSpark at `--max-model-len 1048576`.**
- **Vision-Exp is the only supported model.**

<!-- launcher hashes, maintained by tools/check-current.sh --write -->
sha256 5da63670fcff7f7ac6d5397768a61e878bd11f604acbe7054f1d763ca4e70e8a  launchers/ds4-vision-tp2.sh
sha256 d8db4e958800389585f3c9c897084649cfc22e51e5d01524635f942bd2d3fe21  launchers/ds4-vision-tp4.sh
