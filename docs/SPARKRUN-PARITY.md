# sparkrun ↔ launcher parity

sparkrun is how this repo executes. The shell launchers stay as the manual, fleet-local path and
are the **reference**: every `sparkrun/*.yaml` recipe must match its launcher's runtime, serve argv
and environment exactly, except where a deviation is listed below with a reason. To *operate* the
launcher side rather than compare it, see [`LEGACY-LAUNCHERS.md`](LEGACY-LAUNCHERS.md).

| Launcher (reference) | Recipe (primary) |
|---|---|
| [`../scripts/launch/ds4-vision-tp2.sh`](../scripts/launch/ds4-vision-tp2.sh) | [`../sparkrun/ds4-vision-exp-tp2.yaml`](../sparkrun/ds4-vision-exp-tp2.yaml) |
| [`../scripts/launch/ds4-vision-tp4.sh`](../scripts/launch/ds4-vision-tp4.sh) | [`../sparkrun/ds4-vision-exp-tp4.yaml`](../sparkrun/ds4-vision-exp-tp4.yaml) |

This parity is **machine-checked**, not aspirational: `scripts/check/test-prompt-token-details.py`
parses both sides and fails CI on drift. What it asserts is listed at the bottom.

---

## 1. How the runtime is reproduced without building an image

The launcher runs a prebuilt local image and bind-mounts the patched files over it. sparkrun
cannot `docker build` and cannot distribute an unpublished local image, so
[`../scripts/build/stage-dspark-runtime.sh`](../scripts/build/stage-dspark-runtime.sh)
performs the same mutations **inside the container** during `pre_exec`, from a pinned tarball of
this repo. One script defines the payload for every recipe, so a patch bump is one edit.

| # | Mutation | Source in this repo | Verified by |
|---|---|---|---|
| 1 | `kill(1)` shim (base image ships none; teardown calls it) | created inline | present before serve |
| 2 | `cp -a recipe/overlay/vllm/. → site-packages/vllm/` — carries Patches 1, 2, 2b, 3, 4, 6 (#30) | `recipe/overlay/vllm/` | grep `is_prefill_chunk` (Patch 3), `shared_experts.gate_up_proj` (Patch 4), `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS` (Patch 6 #54) — each **fatal** if absent |
| 3 | NVFP4 stage A → B → C patch (makes `nvfp4_ds_mla` a valid `--kv-cache-dtype`) | `recipe/nvfp4/Dockerfile.stage-{a,b,c}` heredocs, sed-extracted and piped to the interpreter | the dtype is accepted at serve time; stage files must hold exactly one `PY` heredoc each |
| 4 | Vision port: `patch_vision.py` → `models/deepseek_v4/nvidia/model.py` | `vision-exp/port/patch_vision.py` | grep `aligner` in patched `model.py` — **fatal** if absent |
| 5 | Multimodal registry alias `DeepseekV4VForConditionalGeneration` | `vision-exp/port/patch_registry.py` | grep in `registry.py` **plus** a real `import` assert on `_MULTIMODAL_MODELS` |
| 6 | `ds4v_vision.py`, `ds4v_mm.py` installed next to `model.py` | `vision-exp/port/` | `py_compile` + the import check pulls both |
| 7 | Clear `$VLLM_CACHE_ROOT/modelinfos` | — | without it the alias reuses the stale text-only entry → "is not a multimodal model" |
| 8 | torch `_functorch` AOTAutogradCache → own subdir | patched in place (anchor asserted once) | `py_compile`; prevents `FileExistsError` on shared JIT caches |

Idempotency markers (`.staged-dspark-overlay-<pin>`, `.staged-ds4v-port-<pin>`) are keyed to
the payload pin, so a re-run with a new pin re-applies rather than skipping. Patch semantics,
history and per-patch verification live in [`PATCHES.md`](PATCHES.md) — not repeated here.

## 2. Serve argv parity

Everything the launcher passes, the recipe passes. The recipe spells values as `{defaults}` so
`sparkrun show`/`-o` can see them; the resolved argv is what the test compares.

| Flag | TP2 | TP4 | Notes |
|---|---|---|---|
| `--hf-overrides` | `{"architectures":["DeepseekV4VForConditionalGeneration"]}` | same | Required for image input; selects the registry alias |
| `--served-model-name` | `deepseek-v4-flash-dspark` | same | **Match deliberately.** A sparkrun client and a launcher client must see one name |
| `--host` / `--port` | `0.0.0.0` / `8888` | same | |
| `--trust-remote-code` | ✓ | ✓ | |
| `--tensor-parallel-size` / `--pipeline-parallel-size` | 2 / 1 | 4 / 1 | On Spark, TP = node count |
| `--kv-cache-dtype` | `nvfp4_ds_mla` | same | Stage C padded envelope — [`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) |
| `--block-size` | 256 | 256 | |
| `--max-model-len` | 1048576 | same | |
| `--max-num-seqs` | 12 | 64 | |
| `--max-num-batched-tokens` | 8192 | 8192 | |
| `--max-cudagraph-capture-size` | **12** = 2×(1+k) | **66** = 11×(1+k) | Must be a multiple of `1+k` **and** ≥ `--max-num-seqs` — [`LAUNCH-FLAGS.md`](LAUNCH-FLAGS.md) |
| `--gpu-memory-utilization` | 0.85 | 0.85 | |
| `--enable-prefix-caching` | ✓ | ✓ | |
| `--enable-prompt-tokens-details` | ✓ | ✓ | Cached-token readout — [`CACHE-REPORTING.md`](CACHE-REPORTING.md) |
| `--async-scheduling` | ✓ | ✓ | |
| `--enable-chunked-prefill` | ✓ | ✓ | |
| `--speculative-config` | dspark, k=5, probabilistic | same | |
| `--tokenizer-mode` | `deepseek_v4` | same | Makes `chat_template.jinja` inert |
| `--distributed-executor-backend` | `mp` | `mp` | |
| `--tool-call-parser` + `--enable-auto-tool-choice` | `deepseek_v4` | same | |
| `--reasoning-parser` / `--reasoning-config` | `deepseek_v4` + `think` markers | same | |
| `--default-chat-template-kwargs` | `{"thinking":true}` | same | Thinking is **on** by default |
| `--generation-config` | `vllm` | same | And **no** `--override-generation-config` (repetition penalty is a crash on this path) |
| `--enable-flashinfer-autotune` | ✓ | ✓ | |

Cluster flags the launcher passes by hand (`--nnodes`, `--node-rank`, `--master-addr`,
`--master-port`, `--headless`) are appended by sparkrun's `vllm-distributed` runtime and are
deliberately **absent** from the recipe.

## 3. Environment parity

Grouped; each value in the recipe matches the launcher byte-for-byte unless the row says
otherwise. Rationale for each knob: [`LAUNCH-FLAGS.md`](LAUNCH-FLAGS.md) and
[`.env.dspark.example`](../.env.dspark.example).

| Group | Vars | Status |
|---|---|---|
| Node-local JIT caches | `VLLM_CACHE_ROOT`, `DG_JIT_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `TILELANG_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR`, `TILELANG_CLEANUP_TEMP_FILES`, `DG_JIT_USE_NVRTC`, `DG_JIT_NVCC_COMPILER` | **Same values, different path**: launcher `/vllm-cache/…` ↔ recipe `/cache/runtime/…` (sparkrun's per-host runtime cache). Same rule: never share these between ranks (deviation 3) |
| Timeouts | `VLLM_ENGINE_READY_TIMEOUT_S=3600`, `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` | identical |
| Model/attention | `VLLM_ALLOW_LONG_MAX_MODEL_LEN`, `VLLM_TRITON_MLA_SPARSE`, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS`, `VLLM_SKIP_INIT_MEMORY_CHECK`, `VLLM_USE_FLASHINFER_SAMPLER` | identical |
| B12X MoE | `VLLM_USE_B12X_MOE`, `VLLM_USE_B12X_WO_PROJECTION`, `VLLM_B12X_W4A16_FORCE_BLOCKS_PER_SM`, `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M`, `B12X_W4A16_TC_DECODE`, `VLLM_DSV4_B12X_COMPRESSED_MLA` | identical |
| DSpark | the nine `VLLM_DSPARK_*` + `VLLM_DSV4_DSPARK_*` + `DSPARK_SLOT_CLAMP` | identical |
| Patch 6 (#54) tuning | `VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION=0.30`, `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS=1` | identical to the launcher defaults (the launcher reads `PROTECTED_FRACTION`/`SWA_RECYCLE` from the shell; the recipe pins the same numbers) |
| Arch pins | `TORCH_CUDA_ARCH_LIST`, `FLASHINFER_CUDA_ARCH_LIST`, `FLASHINFER_DISABLE_VERSION_CHECK`, `PYTORCH_CUDA_ALLOC_CONF` | identical |
| Fabric (portable half) | `NCCL_NET`, `NCCL_IB_DISABLE`, `NCCL_CROSS_NIC`, `NCCL_IB_MERGE_NICS`, `NCCL_CUMEM_ENABLE`, `NCCL_IGNORE_CPU_AFFINITY`, `NCCL_DEBUG`, `NCCL_NVLS_ENABLE` | identical |
| Fabric (host-specific) | `NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `TP_SOCKET_IFNAME`, `NCCL_IB_GID_INDEX`, `VLLM_HOST_IP` | **Delegated to sparkrun's per-host detection** — deviation 4 |
| Offline model serving | `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `HF_HOME` | **Not set** — sparkrun distributes the checkpoint itself (deviation 5) |

## 4. Container/config parity

| Setting | Launcher | Recipe | Status |
|---|---|---|---|
| `--network` | `host` | `host` | identical |
| `--shm-size` | `64g` | `64gb` | identical intent |
| `--ulimit memlock` | `-1:-1` | `-1:-1` | identical |
| `--ulimit stack` | `67108864` | `67108864` | identical |
| `--device /dev/infiniband` | ✓ | ✓ | identical |
| `--restart` | `no` (container stays for inspection) | `restart_policy: "no"` | identical (also forces `auto_remove: false`) |
| `--privileged` | not passed | `privileged: false` | identical, and stated explicitly so sparkrun's `true` default cannot widen the container |
| `--ipc` | `host` | `shareable` | **deviation 1** |
| GPU access | `--gpus all` | platform default (DGX Spark pins `gpus: all`) | identical |

## 5. Deliberate deviations

1. **`ipc: shareable`, not `host`.** sparkrun runs the container as the SSH user. Under a host
   IPC namespace, every POSIX semaphore the workload creates is a host file owned by a
   non-privileged UID, and `systemd-logind`'s `RemoveIPC=yes` (DGX OS default) reaps it ~10 s
   after the launching session closes — which kills long-starting engines, because sparkrun
   launches detached. A container-private namespace of the same size needs nothing from the
   host and is immune. Same 64 GB budget, no host coupling.
2. **Image source.** The fleet runs
   `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b`, which is local and
   unpublished. The recipe pins the public base by **digest** and stages the patch set in the
   container. To run the fleet image instead: `sparkrun run <recipe> --image <ref>`. Verify
   equivalence with the version check in §6.
3. **Cache path.** `/cache/runtime` (sparkrun's per-host runtime cache) instead of the
   launcher's dedicated `/vllm-cache` bind. The invariant that matters — node-local, never
   NFS-shared between ranks — is preserved. `runtime_cache: false`, or repoint the seven vars,
   if `/cache/runtime` is not node-local on your fleet.
4. **Host-specific NCCL is delegated.** sparkrun detects each host's IB/roce devices and IPs.
   The launcher hardcodes this fleet's plane-A-only topology, because plane B is not on a common
   subnet between those two nodes. If detection lands somewhere unusable, pin it:
   `-o env.NCCL_IB_HCA=rocep1s0f0` (and the three `*_SOCKET_IFNAME`s).
   The launcher also pins `NCCL_IB_GID_INDEX=3`, which
   [`.env.dspark.example`](../.env.dspark.example) explicitly warns against (GID table indexes
   drift — issue #38). The recipe follows the warning and does not pin it.
5. **Checkpoint source.** The launcher bind-mounts pre-staged weights at `/models/<MODEL_DIR>`
   and runs fully offline. The recipe pins the HF repo id **and revision**
   (`86f746b3…`, the `CURRENT.md` pin) and lets sparkrun distribute it. For pre-placed weights,
   override with an absolute path (`-o model=/mnt/bluey-models/DeepSeek-V4-Flash-Vision-Exp`) —
   sparkrun identity-mounts it and skips the download.
6. **No `post_exec` / `post_commands`.** sparkrun v0.2.40 runs post hooks before it starts
   following logs, gated on a hardcoded ~4-minute port-readiness timeout. This model needs
   ~5–6 min to open the port, so any post hook reports `Server port never became ready` against
   a cluster that booted fine. Use `scripts/bench/speedtest-starfall.sh` instead.
7. **The two launchers are not symmetric with each other, and each recipe mirrors its own
   launcher rather than the other one.**

   | knob | TP2 launcher | TP4 launcher |
   |---|---|---|
   | `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | set, `${...:-1800}` | **not set** |
   | `MTP_NUM_TOKENS` | **not set** | set to `5` |
   | `DSPARK_SLOT_CLAMP` | `1` | `1` |

   The recipes reproduce exactly what each launcher passes, so parity is against the launcher you
   are replacing, not against a merged union. Two of these look accidental rather than deliberate
   — both tiers run the same `k=5` probabilistic DSpark spec, so it is not obvious why only TP4
   sets `MTP_NUM_TOKENS` or why only TP2 raises the execute-model timeout. Worth confirming
   against the fleet before either is treated as intentional; changing a launcher means
   re-measuring, so it is left as-is here.

## 6. Verify a sparkrun boot

```bash
# 1. Patch set actually landed (pre_exec already fail-fasts on all three)
sparkrun logs ds4-vision-exp-tp2 | grep -E "stage-runtime|FATAL"

# 2. The runtime is the one this repo is measured on
docker exec <container> /opt/env/bin/python -c "import vllm; print(vllm.__version__)"   # expect 0.21.1rc1.dev339+g1967a5627bc3

# 3. B12X is live — a missing line means the half-speed fallback
sparkrun logs ds4-vision-exp-tp2 | grep "Using 'B12X' Mxfp4 MoE backend"

# 4. Served id and context
curl -fsS http://<head>:8888/v1/models   # deepseek-v4-flash-dspark, max_model_len 1048576

# 5. Same preflight the launcher path uses, on every rank
./scripts/check/check-patch4.sh <head-container> <worker-container>

# 6. Multimodal path actually answers
#    send one image_url request; "is not a multimodal model" = port/alias/modelinfos gap
```

KV pool size is a per-boot figure — read it off the boot you are quoting
([`BENCHMARKS.md`](BENCHMARKS.md)).

## 7. Intentionally not replicated

| Thing | Why not |
|---|---|
| Patch 5 (stop strings inside reasoning) | Optional and off by default. Thinking **is** the default here, so a harness that sends its own `stop` sequences is exposed — mount it if you run one. [`PATCHES.md`](PATCHES.md) |
| Patch A (drafter capture sizes) | Optional, off by default, needs `VLLM_DSPARK_DRAFT_CAPTURE_SIZES` on every rank |
| `scripts/experimental/*` (anemll lane) | Different runtime entirely; not the golden recipe |
| Drop-the-page-cache-before-launch | A host-level step, not a workload property. Still required — [`CURRENT.md`](../CURRENT.md) |

## 8. Changing the payload pin

`defaults.ds4_src_commit` in each recipe is the commit of this repo the in-container staging
fetches. Bump it when the overlay, NVFP4 stages or the vision port change, then re-run the
checklist in §6. Ad-hoc override without editing: `-o ds4_src_commit=<sha>`.

Both recipes carry the same pin; `test_recipe_payload_pins_match` fails CI if they drift apart.

## 9. What CI enforces

From `scripts/check/test-prompt-token-details.py`:

- every launcher and every recipe passes `--enable-prompt-tokens-details` exactly once, and
  neither mistypes it;
- `--max-cudagraph-capture-size % (1+k) == 0` and `>= --max-num-seqs`, with `k` read from the
  spec on each side;
- the spec stays `dspark` + `probabilistic`;
- **launcher↔recipe argv parity**: every launcher flag is present in the matching recipe, and
  the only launcher-only flags allowed are the cluster flags sparkrun appends itself;
- every recipe value that the launcher passes as a literal resolves equal through `defaults`;
- the two recipes pin the same `ds4_src_commit`.
