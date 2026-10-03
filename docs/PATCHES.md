# Patch Reference — DeepSeek-V4-Flash-Vision-Exp on DGX Spark

One reference for every source patch this recipe carries: what it fixes, where its source
lives, where it is staged at runtime, how it reaches the container, whether the current
launchers require it, and how to prove it landed on **every** node.

Related: [`CURRENT.md`](../CURRENT.md) (live deployment state and pins) ·
[`../vision-exp/README.md`](../vision-exp/README.md) (vision port internals) ·
[`CREDITS.md`](../CREDITS.md) (provenance of the concurrency work behind Patches 1/2) ·
[`BENCHMARKS.md`](BENCHMARKS.md) (how to read a number) ·
[`UPGRADE-OFFICIAL-MAIN.md`](UPGRADE-OFFICIAL-MAIN.md) (moving to stock vLLM main)

Paths in the tables are relative to the repo root. Container-side target paths are under
`/opt/env/lib/python3.12/site-packages/vllm/` on this recipe's image — other images use
different roots (see [Staging and verification](#staging-and-verification)).

## Patch inventory

| Patch | What it fixes | Source in this repo | Staged at | Delivery | Required by current launchers | How to verify it landed |
|---|---|---|---|---|---|---|
| **1** | Draft acceptance collapses toward 0 at `--max-num-seqs > 1`: persistent DSpark draft KV is keyed by vLLM batch-row position, which continuous batching condenses | `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`, `recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark.py`, `recipe/overlay/vllm/v1/worker/gpu_model_runner.py` | — | Baked into the runtime image | Required **in the image**; no bind mount, no preflight | `grep -c _req_id_to_slot "$VLLM_ROOT/v1/spec_decode/dspark_proposer.py"` → ≥ 1 |
| **2** | HTTP 500 (`got 41 rows for batch_size=2`) on real staggered arrivals: rectangular `[batch, seq, H]` view cannot represent mixed prefill/decode steps | `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`, `recipe/overlay/vllm/models/deepseek_v4/nvidia/dspark.py` | — | Baked into the runtime image | Required **in the image**; needs `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` | `grep -c _store_main_kv_ragged "$VLLM_ROOT/models/deepseek_v4/nvidia/dspark.py"` → ≥ 1 |
| **2b** | 500 on prefill-heavy steps with **no** rejection (`got 166 rows for batch_size=3`): ragged detection was gated on rejection | `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py` | — | Baked into the runtime image | Required **in the image** | `grep -n "ragged = len(set(seg_lengths))" "$VLLM_ROOT/v1/spec_decode/dspark_proposer.py"` → present inside the `_gpu_rejected_context_mask` branch |
| **3** | Cold-start garble on long resumed conversations (prompt echo, leaked tool/schema text): spec-token placeholders resized on chunked-prefill chunks | `recipe/overlay/vllm/v1/core/sched/scheduler.py` | `/var/tmp/patch3-scheduler.py` | Bind mount **and** baked in the overlay image | **Yes** — launcher exits `4` if the staged file is missing | [`../scripts/check-patch3.sh`](../scripts/check-patch3.sh) (both nodes) |
| **4** | DSpark draft's always-on shared expert loads **uninitialised** → ~half decode speed with perfect output quality, silently | `recipe/overlay/vllm/v1/spec_decode/dspark.py` (diff: `patches/0004-dspark-shared-expert-gate-up-proj.patch`) | `/var/tmp/spec-dspark.py` | Bind mount (also baked in the overlay image) | **Yes** — launcher exits `4` if the staged file is missing | [`../scripts/check-patch4.sh`](../scripts/check-patch4.sh) (both nodes) |
| **5** | Client `stop` strings fire inside the reasoning segment → `content: null` on thinking requests from harnesses that send stops | `patches/0005-suppress-stops-in-reasoning.patch`, port: `patches/0005-port-vllm-0.25-apply.py` | operator-chosen (e.g. `/var/tmp/detokenizer.py`) | Bind mount / patch-at-start only | **No** — not in the current launcher mount set | grep the patched `v1/engine/detokenizer.py` for the guard markers (`PATCH(stop-in-reasoning)`) |
| **6 (upstream #30)** | `AttributeError: 'ShmRingBuffer' object has no attribute 'shared_memory'` after a multi-minute model load: the scheduler-output queue's SHM name is unlinked before the late reader opens it | `recipe/overlay/vllm/distributed/device_communicators/shm_broadcast.py`, `recipe/overlay/vllm/v1/executor/multiproc_executor.py` | — | Baked into the runtime image | Not bind-mounted; carried by the overlay image | Assert `MessageQueue.preopen_from_handle` exists in the container; overlay build runs the CPU-only lifetime regression |
| **6 (upstream #54)** | Prefix cache lost on long conversations (235 s re-prefill of ~400K tokens): unbounded prompt-block protection + sliding-window LRU churn | `recipe/overlay/vllm/v1/core/single_type_kv_cache_manager.py` (diff: `patches/0006-kv-cache-prompt-protection-cap-and-swa-recycle.patch`) | `/var/tmp/patch6-single_type_kv_cache_manager.py` | Bind mount (not `COPY`ed by the overlay Dockerfile) | **Yes** — launcher exits `4` if the staged file is missing | `grep -c VLLM_SWA_RECYCLE_SKIPPED_BLOCKS "$VLLM_ROOT/v1/core/single_type_kv_cache_manager.py"` → ≥ 1 |
| **A (optional)** | The proposer shares the target's cudagraph capture sizes: they round to multiples of `1+k`, so a batch-1 draft dispatches on the 6-bucket and the draft MoE processes 20 draft tokens/step for one stream instead of 5 | `patches/A-drafter-sizes/v1/spec_decode/dspark_proposer.py` | operator-chosen (e.g. `/var/tmp/dspark_proposer.py`) | Bind mount + `VLLM_DSPARK_DRAFT_CAPTURE_SIZES` on **both** ranks | **No** — optional, off by default, not in the current launcher mount set | `grep -c VLLM_DSPARK_DRAFT_CAPTURE_SIZES "$VLLM_ROOT/v1/spec_decode/dspark_proposer.py"` → ≥ 1, and the same boot-log line on both ranks |
| **Vision port** | Native image input: stock `DeepseekV4ForCausalLM` has no vision tower/aligner ("no module or parameter named `aligner`"), and the registry's static arch table says the model "is not a multimodal model" | `vision-exp/port/*` generated per image by [`../vision-exp/build-ds4v-files.sh`](../vision-exp/build-ds4v-files.sh) | `/var/tmp/ds4v_model.py`, `/var/tmp/ds4v_vision.py`, `/var/tmp/ds4v_mm.py`, `/var/tmp/ds4v_registry.py` | Bind mount (derived per image) | **Yes** — TP4 preflights all four; TP2 fails at container start | build-script marker asserts + `grep -c DeepseekV4VForConditionalGeneration "$VLLM_ROOT/model_executor/models/registry.py"` |

## The two "Patch 6"es

Two independent upstream fixes both carry the **Patch 6** label, and this fork adopts both:

- **Patch 6 (upstream #30)** — the scheduler-queue **SHM lifetime** fix, adopted from
  [PR #30](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/30).
- **Patch 6 (upstream #54)** — the **KV prefix-cache** fix (prompt-protection cap + sliding-window
  page recycling), adopted from
  [PR #54](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/54).

Always disambiguate with the PR number. Upstream file names are kept exactly as shipped
(`patch6-single_type_kv_cache_manager.py`,
`patches/0006-kv-cache-prompt-protection-cap-and-swa-recycle.patch`, the `Patch 6` anchors in
`sparkrun/`), so those paths and anchors still resolve.

One more collision to know: inside the `0006-kv-cache-...` diff, the two halves of the KV fix
are internally labelled "Patch 5" (protection cap) and "Patch 6" (recycling). That is **not**
this file's Patch 5 (stop strings). Below, the two halves are called **cap** and **recycle**.

## Patch 1 — request-stable KV slot

### Symptom

At `max_num_seqs > 1`, draft acceptance collapsed toward 0 (garbage drafts) with no crash —
the engine silently degraded to single-stream quality.

### Root cause

DSpark's draft keeps one persistent cross-step tensor per attention module —
`DeepSeekV4DSparkAttention.main_kv_cache`, shape `[max_num_seqs, window, head_dim]` — a
per-row **ring buffer** holding each sequence's sliding-window KV history. It was read and
written by **batch-row position** (`main_kv_cache[:batch_size]`), and the draft proposer
carried **no request identity**.

Under vLLM-v1 continuous batching the running set is *condensed* whenever a request finishes:
a later request is moved into the freed row. The model's persistent `main_kv_cache` row is
**not** moved with it, so after a condense a request reads a ring buffer belonging to a
**different** request → corrupted draft context → acceptance collapse. Single-stream never
condenses row 0, which is why it always looked fine.

### Fix

Key the persistent cache by a **stable per-request slot** instead of the batch row:

- `v1/spec_decode/dspark_proposer.py` — add `self._req_id_to_slot: dict[str, int]` and
  `self._free_slots`. `_row_to_slot(req_ids)` reclaims slots of finished requests, assigns a
  free slot (lowest-first) to new ones, and returns the slot per row in `req_ids` order. A
  persistent, cudagraph-captured `_draft_slot_index_buffer` carries the slots into the graphed
  draft read path.
- `models/deepseek_v4/nvidia/dspark.py` — `store_main_kv` and `forward_dspark` index the
  cache by `slot_index` (gather via `index_select` on read, scatter via `index_copy_` on write)
  instead of `[:batch_size]`.
- `v1/worker/gpu_model_runner.py` — pass `req_ids=self.input_batch.req_ids` into `propose()`
  (only for the DSpark proposer).

### Safety

The math is unchanged; only the physical row a request uses is re-routed. When the computed
permutation is identity (a server that only ever runs one request at a time keeps slot 0), the
code takes the **original in-place write path, byte-for-byte**. Gating is on the *permutation
identity*, not on `batch == 1`, so "the batch condenses to one surviving request holding a
non-zero slot" stays correct.

### Verification

Patches 1/2/2b have no dedicated check script; grep the running container
(`VLLM_ROOT` resolution is documented under
[Staging and verification](#staging-and-verification)):

```bash
docker exec <container> grep -c _req_id_to_slot "$VLLM_ROOT/v1/spec_decode/dspark_proposer.py"       # >= 1
docker exec <container> grep -c "def _row_to_slot" "$VLLM_ROOT/v1/spec_decode/dspark_proposer.py"   # >= 1
docker exec <container> grep -c _draft_slot_index_buffer "$VLLM_ROOT/v1/spec_decode/dspark_proposer.py" # >= 1
```

Behaviourally: hold two or more concurrent requests through a finish/condense cycle and confirm
per-position acceptance does not collapse toward 0.

### Configuration

No dedicated switch. `DSPARK_SLOT_CLAMP` (default `1`, set by both launchers) bounds-checks
and clamps out-of-range `slot_index` values at the gather sites before they can trip a
device-side `indexSelectSmallIndex` assert; `DSPARK_SLOT_CLAMP=0` reverts to detect-and-log
only, as an A/B kill switch for whether the clamp is load-bearing on a given rig. Related
slot-safety work (canary instrumentation, stale draft-KV slot clamp) is credited to
@paulbrav in [`CREDITS.md`](../CREDITS.md).

## Patch 2 — ragged context path (`query_start_loc`)

### Symptom

Under real (independent / staggered) arrivals at `max_num_seqs > 1`, the server returned
HTTP 500:

```text
ValueError: DSpark currently requires uniform flattened per-request inputs;
got 41 rows for batch_size=2.   (dspark_proposer.py: _view_by_request)
```

### Root cause

`prepare_context` reshaped the flat target hidden states into a **rectangular**
`[batch, seq, H]` via `_view_by_request` / `_positions_by_request`, asserting every request
contributed the **same** number of rows. With chunked prefill (required — disabling it needs
`max_num_batched_tokens >= max_model_len`, infeasible at long context) a single step **mixes
prefill and decode** rows, so per-request row counts differ: "41 rows for `batch_size=2`" is
one request prefilling alongside one decoding. A rectangular reshape is impossible → crash.
The static benchmark passed only because all prompts were identical length (uniform).

### Fix

Make the context path **ragged** using `query_start_loc` (per-request segment offsets) — the
same mechanism `_trim_rejected_target_context` already used:

- `dspark_proposer.py` `prepare_context` detects non-uniform segment lengths
  (`ragged = len(set(seg_lengths)) != 1`). In the ragged branch it skips the rectangular view,
  computes each request's draft anchor with a flat index
  `anchor_idx = starts + clamp(len - rejected - 1, 0, len - 1)`, `index_select`s the
  per-request last hidden/positions, and passes the flat hidden plus `query_start_loc` plus
  `slot_index` to `prefill_main`.
- `models/deepseek_v4/nvidia/dspark.py` dispatches `store_main_kv(..., query_start_loc=...)`
  to `_store_main_kv_ragged`, which loops requests over `query_start_loc`, truncates each
  segment to the last `window_size` rows, computes `slots = positions % window`, applies the
  rejected-suffix mask, and `index_copy_`s into that request's slot. `prefill_main` threads
  `query_start_loc` through and skips the rectangular view in ragged mode.

### Safety

Storage is **position-addressed** (`positions % window`), so it never needed uniform lengths —
only the intermediate rectangular view did. When lengths are uniform
(`query_start_loc is None`, static or single-stream) the original rectangular fast path runs
unchanged. Ragged/mixed steps run **eager** (mixed steps are never cudagraph-captured), so
dynamic Python loops and variable shapes are safe; the uniform decode-only graphed path is
untouched.

### Scope

Only the `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` path was made ragged — the path used in
serving. The legacy `_trim_rejected_target_context` path still assumes uniform input. **Run
with `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.**

### Verification

```bash
# Ragged store path present in the running container
docker exec <container> grep -c _store_main_kv_ragged \
  "$VLLM_ROOT/models/deepseek_v4/nvidia/dspark.py"          # >= 1

# Ragged dispatch on the read side
docker exec <container> grep -n "if query_start_loc is not None" \
  "$VLLM_ROOT/models/deepseek_v4/nvidia/dspark.py"          # present in store_main_kv
```

If the launcher's `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` is not `1`, the ragged path is never
taken even when the code is present.

### Configuration

| Setting | Required value |
|---|---|
| `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` | `1` (set by both launchers) — the only path made ragged |
| `--enable-chunked-prefill` | on; without it long-context uniform batches are infeasible |

For images **other** than this recipe's overlay build, do not copy this repo's proposer over
them — a `propose()` signature mismatch (`got an unexpected keyword argument 'req_ids'`) is the
usual result. Use [`../scripts/apply-nonuniform-guard.py`](../scripts/apply-nonuniform-guard.py),
which patches the target image's own proposer in place so the guard is always
signature-correct.

## Patch 2b — ragged detection independent of rejection

### Symptom

After Patch 2, a prefill-heavy step with **no rejection** still 500'd:

```text
ValueError: ... got 166 rows for batch_size=3     (_view_by_request)
```

Earlier staggered tests with uniform-ish prompts missed it; GSM8K's varied prompt lengths hit it.

### Root cause

Patch 2 computed `ragged` **only inside** `if gpu_mask and num_rejected_tokens_gpu is not None`.
On steps with no rejection (`num_rejected` is `None`, e.g. fresh requests prefilling) detection
was skipped and the code fell through to the rectangular `_view_by_request` → crash. Raggedness
depends on `query_start_loc` segment lengths, **not** on rejection.

### Fix

- Enter the detection/ragged branch whenever `_gpu_rejected_context_mask` is on,
  **regardless of `num_rejected_tokens_gpu`**, which may be `None`.
- In the ragged anchor, default `rejected` to zeros when `num_rejected_tokens_gpu is None`;
  `_store_main_kv_ragged` already handled `None` (no masking).

### Safety

Uniform batches still take the rectangular fast path. The change only removes the path that
fell through to an unsafe reshape; the no-rejection case builds a zero rejected-token tensor
instead of relying on a uniform view. This is the check listed first in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md) for gibberish/loop/CJK-drift symptoms under
concurrency: ragged `query_start_loc` handling must not depend on `num_rejected_tokens_gpu`.

### Verification

```bash
# Ragged flag is computed from segment lengths, inside the mask branch — not inside a
# "num_rejected is not None" branch.
docker exec <container> grep -n "ragged = len(set(seg_lengths))" \
  "$VLLM_ROOT/v1/spec_decode/dspark_proposer.py"
```

### Configuration

Same as Patch 2: `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`.

## Patch 3 — no spec placeholders on prefill chunks (cold-start garble root fix)

**Credit: @roady001 (issue #3)**, who reported that the earlier launch/config change only
reduced the symptom and did not address the root cause, then independently validated this fix on
his own 2× DGX Spark with none of the config workarounds applied — i.e. this is the actual root
cause. Root-cause analysis and patch: **Fable** (see [`CREDITS.md`](../CREDITS.md)).

### Symptom

Resuming an existing long conversation **cold** (server restarted, or the prompt fell out of the
prefix cache) produced garbage at the **start** of the reply — prompt echo, leaked tool/schema
text, "your message was cut off"-style replies — then generation recovered. Warm continuations
of the same conversation were clean. Reproduced deterministically with a ~98K-token prompt at
`temperature 0`: the cold run echoed prompt content from ≈ `prompt_len - 8192` (the last chunk
boundary) and looped; the immediately repeated (warm) run answered correctly.

### Root cause

The DSpark async-scheduling addition in `Scheduler.update_from_output` resizes the `[-1]`
spec-token placeholder list to DSpark's confidence-scheduled draft length. It ran for **every
running, non-stopped request in the batch — including requests still mid-way through a chunked
prefill**. Upstream never installs spec placeholders on prefill chunks:
`AsyncScheduler._update_after_schedule` skips `request.is_prefill_chunk`, and
`Scheduler.update_draft_token_ids` explicitly clears drafts for prefill chunks.

With a long cold prompt (multiple `max_num_batched_tokens=8192` chunks), the illegal
placeholders made the scheduler attach `num_speculative_tokens` spec tokens to the request's
**final prompt chunk**. Two corruptions follow:

1. The drafts verified there were proposed from the **truncated** prompt (mid-prefill DSpark
   drafts — prompt-continuation predictions).
2. The request was in the previous step's **discard set** (mid-prefill), so the worker excludes
   it from `prev_req_id_to_index`; `_prepare_input_ids` then never scatters real draft ids over
   the `-1` placeholders, and the target forward for the final chunk sees invalid token ids at
   the spec positions.

Either way the first sampled tokens of the reply are conditioned on a corrupted prompt tail —
visible as prompt echo / garble at the start of a cold resume, recovering once pure decode
steps take over. Patches 1/2/2b fixed the persistent-KV side, leaving only this cold-start
window.

### Fix

`vllm/v1/core/sched/scheduler.py` (`update_from_output`): only resize spec placeholders for
requests the AsyncScheduler itself would give placeholders — `new_token_ids` non-empty,
`not request.is_prefill_chunk`, and `request.status == RequestStatus.RUNNING` (a preempted
request must keep its cleared spec list).

### Verification

[`../scripts/check-patch3.sh`](../scripts/check-patch3.sh) is a fail-closed preflight; run it
against **every** node's container:

```bash
./scripts/check-patch3.sh <head-container> <worker-container>
# exit 0 = present everywhere, exit 1 = missing somewhere, exit 2 = usage error
```

It asserts `is_prefill_chunk` appears in the container's `v1/core/sched/scheduler.py` and
prints the resolved vLLM root. A pre-Patch-3 image (`probe-c-p2b`) boots clean, passes smoke
tests and serves warm requests correctly — it only garbles on **cold** prefill, so a warm
5-prompt gate passes on a broken deployment. Measured on 2× DGX Spark with a ~20k-token agent
prompt forced cold: without Patch 3, 44/44 requests garbled across `k=3`, `k=5` and all
documented settings; with Patch 3, 0/28 garbled. Behavioural check: the ~98K-token cold-resume
prompt at `temperature 0` produces a correct first reply matching the warm rerun.

**Mount-path warning:** mount to the path the script prints, not a remembered one — a bind mount
to the wrong path fails **silently**.

### Configuration

- No configuration is needed for the fix itself.
- `repetition_penalty=1.05` is a separate DSpark crash risk (illegal memory access); drop it
  first if an IMA crash appears. The launchers pass `--generation-config vllm` only.
- `draft_sample_method=probabilistic` (both launchers, `k=5`) is a valid tuning option but is
  **not** needed to fix this garble once the scheduler guard is in place.
- Do not read a `k` regression as a Patch 3 problem: the "k=3 wins" A/B was measured **without**
  the Patch 4 mount and is retracted (issue #48). See Patch 4.

## Patch 4 — DSpark draft shared-expert loader fix

Diff: [`patches/0004-dspark-shared-expert-gate-up-proj.patch`](../patches/0004-dspark-shared-expert-gate-up-proj.patch)

### Symptom

Decode runs at roughly **half speed with perfect output quality** — no garble, no drift, no
special-token leakage, and no error. The deficit shows in the spec-decode metrics, not the step
rate: drafted throughput is healthy, accepted throughput collapses, `steps/s` never moves.

### Why it looks like a model regression (and isn't)

`tok/s = steps/s × accepted-tokens-per-step`. Under this bug the step rate is unchanged — the
engine, the fabric, the KV cache and the target model are all healthy. The entire deficit lands
in `accepted-tokens-per-step`.

Because speculative decoding is **verified by the target model**, a bad draft can never corrupt
output — it can only cost speed. So a broken drafter presents as "these weights are slow", which
is exactly the wrong place to look.

### Root cause

The DSpark draft's FFN is a `DeepseekV4MoE` containing a shared expert built as a
`DeepseekV4MLP`, whose projections are `gate_up_proj` (a `MergedColumnParallelLinear`, fed by
checkpoint tensors `w1` and `w3`) and `down_proj` (fed by `w2`). The draft weight loader
renames only `w2`:

```python
# vllm/models/deepseek_v4/nvidia/dspark.py:1066
name = name.replace(".shared_experts.w2", ".shared_experts.down_proj")
```

and the stacked-parameter mapping it consults contains **only the two attention entries**:

```python
# vllm/v1/spec_decode/dspark.py:15-18   (BEFORE)
_STACKED_PARAM_NAME_MAPPING = (
    ("attn.fused_wqa_wkv", ".attn.wq_a", 0),
    ("attn.fused_wqa_wkv", ".attn.wkv", 1),
)
```

So `shared_experts.w1` and `shared_experts.w3` match nothing and fall through to
`params_dict.get(name)`, which drops them via
`logger.debug("Skipping unknown DSpark weight %s")` (`dspark.py:1122-1125`). `logger.debug`
is invisible at the default INFO level, so **the load reports success**.

**12 checkpoint tensors are lost** — `w1` and `w3`, each as `weight` + `weight_scale_inv`
(renamed from `.scale` before the mapping is consulted), across all three draft stages. That
leaves six parameters uninitialised:

```
model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.weight
model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.weight_scale_inv
```

`n_shared_experts: 1`, and the shared expert is **always-on** — its output is summed into every
token, unconditionally, alongside the routed experts. Each of the three draft stages therefore
runs with its always-active expert uninitialised. The drafter still produces fluent, plausible
tokens; they just disagree with the target far more often.

**The tell: the target loader has the rows the draft loader is missing.**

```python
# vllm/models/deepseek_v4/nvidia/model.py:1952-1953   (target — correct)
("gate_up_proj", "w1", 0),
("gate_up_proj", "w3", 1),
```

The draft loader lost them when its mapping was narrowed to avoid a name collision: the DSpark
checkpoint also contains `markov_head.markov_w1`, which must not be treated as an FFN `w1`
shard. The narrowing worked, and took the shared-expert shards with it.

### Fix

Two rows added to `_STACKED_PARAM_NAME_MAPPING`:

```python
("shared_experts.gate_up_proj", ".shared_experts.w1", 0),
("shared_experts.gate_up_proj", ".shared_experts.w3", 1),
```

### Safety

- `map_dspark_stacked_param_name()` returns early on `".experts."` (note the **leading dot**),
  so routed experts (`ffn.experts.0.w1`) are untouched and still go through `expert_mapping`.
- The new patterns are anchored on the full `".shared_experts.wN"` segment. `markov_w1` has no
  `.shared_experts.` prefix and cannot match. Verified against the mapper:

  ```
  43.ffn.shared_experts.w1.weight            -> (...shared_experts.gate_up_proj.weight, 0)
  43.ffn.shared_experts.w1.weight_scale_inv  -> (...shared_experts.gate_up_proj.weight_scale_inv, 0)
  44.ffn.shared_experts.w3.weight            -> (...shared_experts.gate_up_proj.weight, 1)
  43.ffn.experts.0.w1.weight                 -> None      (routed expert, unchanged)
  45.markov_head.markov_w1.weight            -> None      (no collision)
  43.attn.wkv.weight                         -> (...attn.fused_wqa_wkv.weight, 1)
  ```

- `.scale` → `.weight_scale_inv` renaming happens at `dspark.py:1074-1080`, *before* the
  mapping is consulted, and `_EXPERT_SCALE_RE = r"\.experts\.\d+\.w[123]\.scale$"` requires a
  digit after `.experts.` — so `shared_experts` correctly takes the `.weight_scale_inv` branch
  that `MergedColumnParallelLinear` expects for FP8 block quantisation.
- A wrong mapping **fails loudly** — `dspark.py:1086-1087` raises `KeyError` rather than
  skipping — so a clean boot is positive evidence the mapping resolved.

### Verification

With the loader mount as the only variable (2× DGX Spark, TP=2, same runtime, checkpoint and
flags): mean acceptance 25.7% → 60.2%, tokens/step 2.28 → 4.01, mean decode 32.7 → 55.4
tok/s, peak 42.0 → 66.1 tok/s. Warmed, single-stream, temp 0, per content type:

| workload | stock loader | **patched** |
|---|---:|---:|
| count to 100 | 50.7 | **80.1** |
| code | 47.2 | **51.8** |
| prose | 30.4 | **33.2** |

The gain on *count* carries a warm-up component on top of the loader fix, and acceptance on
prose-heavy traffic stays around 25% on this vision variant — which is why code and prose gain
less than count. Patch 4 is a correctness fix regardless of how the per-workload speed is
attributed: the always-on shared expert was loading uninitialised, and now it is not.
Profile-level numbers for the recipe live in [`CURRENT.md`](../CURRENT.md); how to read them is
in [`BENCHMARKS.md`](BENCHMARKS.md).

```bash
# Fail-closed preflight — run against BOTH nodes
./scripts/check-patch4.sh <head-container> <worker-container>
# exit 0 = present everywhere, exit 1 = missing somewhere, exit 2 = usage error
```

The check counts `shared_experts` occurrences in the container's `v1/spec_decode/dspark.py`:
**6 or more** (two mapping rows + comment) = patched; **0** = stock loader. Definitive runtime
check, with `VLLM_LOGGING_LEVEL=DEBUG`:

```bash
docker logs <container> 2>&1 | grep "Skipping unknown DSpark weight.*shared_experts"   # -> empty
```

Checks to run against a model dir and a runtime before concluding you are unaffected:

```bash
# 1. does the checkpoint carry shared-expert draft tensors? (expect 18: w1/w3 = 12, w2 = 6)
python3 - <<'PY'
import json, glob
wm = json.load(open(glob.glob("<MODEL_DIR>/*.index.json")[0]))["weight_map"]
hits = sorted(k for k in wm if "mtp" in k and "shared_experts" in k)
print(len(hits), "shared-expert mtp tensors")
print("\n".join(hits[:6]))
PY

# 2. does the runtime's mapping include them?
grep -A6 '_STACKED_PARAM_NAME_MAPPING' \
  /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py
```

### Ruled out by measurement

Recorded so nobody repeats the search. Every row below is measured, not assumed.

| tried | result |
|---|---|
| `draft_sample_method` greedy vs probabilistic | no change — a **no-op** for DSpark: the proposer never exports draft probs unless `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1` |
| `fp8_ds_mla` vs `nvfp4_ds_mla` KV | no change — KV dtype is a context lever, not an acceptance lever |
| temperature 0 vs 0.7 | no change |
| B12X kernels off (`VLLM_USE_B12X_MOE=0`, `VLLM_USE_B12X_WO_PROJECTION=0`) | **worse** — the step rate drops, then the run crashes |
| dedicated node pair, zero competing traffic | no change — contention was never involved |
| runtime / image build drift | identical vLLM version and identical DSpark env across nodes |
| draft shared-expert tensor names and dtypes | correct — the tensors exist with the expected names, shapes and dtypes; they are simply never loaded |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` / `_SCHEDULER` | inert — `0.0`/`off` means *no gating*, and the confidence head is skipped on the hot path |

### Configuration

No switch. Two operational notes, then the `k` ceiling:

- The mount is **carried per container run**, and a run command can carry every other mount
  while dropping this one — the loss is silent, which is why the launcher preflights it and why
  [`../scripts/check-patch4.sh`](../scripts/check-patch4.sh) is the first troubleshooting step.
  Verify it landed on **every** node (head *and* worker).
- Spec-depth measurements made without this mount are invalid: the "k=3 wins on this
  checkpoint" A/B was an unpatched measurement and is retracted (issue #48). With Patch 4
  mounted, `k=5` (the launcher default) wins.

**Speculative depth: `k ≤ 5`, or a multiple of 5.** A `k=3` comparison run **without** the
Patch 4 mount measures the loader rather than the drafter, because an uninitialised shared
expert collapses acceptance to the loader's own signature. Measured **with Patch 4** mounted on
a second 2× DGX Spark pair (warm, temp 0, 500K context,
[issue #48](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/issues/48)):

| workload | k=5 | k=3 |
|---|---:|---:|
| count to 300 | **85.5 tok/s** (accept 0.974 · 5.88 tok/step) | 64.4 tok/s |
| code | 49.9 | 49.5 |
| prose | 29.5 | 32.1 |

With Patch 4 mounted, `k=5` wins clearly on predictable work and `k=3` keeps a small edge on
prose. `num_nextn_predict_layers: 3` does not imply `k=3`. Counting figures are
draft-acceptance ceilings, not throughput — see [`BENCHMARKS.md`](BENCHMARKS.md).

- `k` is bounded by the **draft block size**, not the draft layer count: `k ≤ 5` on this image,
  or a multiple of 5.
- `k` above the block size crashes at **first generation**, not only at boot. Patching out the
  boot-time divisibility guard (`num_speculative_tokens % n_predict != 0`) yields
  `RuntimeError: The size of tensor a (7) must match the size of tensor b (5)`. The drafter
  emits exactly `dspark_block_size` (5 on this image) tokens per pass and `propose()` calls it
  once; multi-block drafting is not implemented. Raising `k` past the block on
  DeepSeek-V4-Flash needs proposer work, not a flag.
- The guard only fires when `k > n_predict`, which is why `k=3` and `k=4` boot fine. Upstream
  0.25.2 uses the same `>` check but resolves `n_predict` to `num_nextn_predict_layers` rather
  than `dspark_block_size`, so the guard never fires there, and its DSpark speculator sizes the
  draft block from `k`, so no shape mismatch follows (issue #22).
- `k=7` works only where the draft block is at least 7: MiMo-V2.5 DFlash (`block_size` = 8, run
  at `num_speculative_tokens: 7`), GLM-5.2's DSpark speculator (`block_size` = 8, and its own
  `speculators_config` asks for 7 speculative tokens), and the Inkling DSpark preview
  (`dspark_block_size` = `n_predict` = 7). DeepSeek-V4-Flash's draft block is 5, hence `k=5`.

## Patch 5 — stop strings must not fire inside the reasoning segment (optional)

**Credits: @Capicua25x** (original), **extended by @Mcray4** (output-side arming). Ported to
this image from upstream PR #21 (commit `3ba6ee21`). Issue: **#18**.

Files: [`patches/0005-suppress-stops-in-reasoning.patch`](../patches/0005-suppress-stops-in-reasoning.patch)
and the in-place port
[`patches/0005-port-vllm-0.25-apply.py`](../patches/0005-port-vllm-0.25-apply.py).

### Symptom

A harness that sends `stop` sequences (lm-evaluation-harness sends `stop[:4]` on **every**
request) silently loses answers. Generation starts *inside* `<|im_start|>`, chain-of-thought
naturally restates phrases like `Question:`, the stop fires mid-reasoning, `<|im_end|>` never
arrives, and the reasoning parser returns `content: null`. The request looks like a model
failure; it is a serving-layer one. Hosted deployments of the same model are immune because
they scope stops to content.

### Root cause

vLLM's v1 detokenizer matches client stop strings against the **whole output stream**,
including the reasoning segment.

### Fix

Bind-mount a patched `v1/engine/detokenizer.py` (no rebuild), or run the port script at
container start:

```bash
-v /var/tmp/detokenizer.py:/opt/env/lib/python3.12/site-packages/vllm/v1/engine/detokenizer.py:ro
```

The guard is per-request and needs no configuration. It arms in two modes:

1. **Prompt-side** — if the request's last prompt token is `<|im_start|>`, stop strings stay
   dormant until `<|im_end|>` appears in output.
2. **Output-side** — when the request carries stop strings and reasoning markers are
   configured but the prompt does **not** end with the tag (some templates never put the tag in
   the prompt and the model opens `<think>` as its first output token), the guard arms on that
   first output marker. Whether a model opens a think tag at `temp 0` is nondeterministic
   request-to-request, so one direct-answering probe does not prove a deployment unaffected.

On close, stop checking resumes only **past** the end marker (spec decode delivers `k+1` tokens
per update, so the closing chunk carries reasoning tail). EOS and `max_tokens` are unaffected;
non-thinking requests are untouched. Opt out process-wide with
`VLLM_SUPPRESS_STOPS_IN_REASONING=0`.

### Verification

The port script exits `0` always (a hard exit would kill the boot) and prints `PATCH5-APPLIED` /
`PATCH5-FAILED`; loudness lives in the post-boot grep, not the exit code:

```bash
docker exec <container> grep -c "PATCH(stop-in-reasoning)" \
  "$VLLM_ROOT/v1/engine/detokenizer.py"        # >= 1 when applied
```

**Both nodes.** The launcher syncs the compose and env files to the worker but **not**
bind-mounted patch files. The file must exist at the same path on the worker too, or it silently
runs unpatched and you get confusing half-fixed results.

### Known side effect: reasoning runaways become more visible, not less

Worth stating so it is not read as a regression. Issue #18 (B) is a reasoning runaway in which
`<|im_end|>` never arrives. Because this patch keeps stops dormant until the end marker appears,
a request in that state now keeps stops dormant for its whole life and runs to `max_tokens` —
where previously a client stop string could cut it short by accident.

That is correct by design: a stop string was never meant to bound reasoning, and a run truncated
by one was returning `content: null` anyway. The practical effect is that (B) shows up as a
full-budget request rather than a short one, so a fleet that applies this patch may see
*reported* token usage on those requests rise. The failure rate does not change; only how long
each failure takes to admit it. If you are measuring (B), stop strings are no longer a confound
in either direction.

### Configuration

| Knob | Default | Effect |
|---|---|---|
| `VLLM_SUPPRESS_STOPS_IN_REASONING` | on (`"1"`) | `0` disables the guard **process-wide** (not per request) |

### Status in this repo

**Optional — not part of the current launcher mount set** (Patch 3 / Patch 4 / Patch 6 plus
the four vision-port files). Apply it only if you serve thinking mode to harnesses that send
stop sequences.

## Patch 6 (upstream #30) — preserve the local scheduler queue across long model loads

Adopted from
[PR #30](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/30).
Sources: `recipe/overlay/vllm/distributed/device_communicators/shm_broadcast.py` and
`recipe/overlay/vllm/v1/executor/multiproc_executor.py`.

### Symptom

On the two-node profile, replacing the stock checkpoint with a large documented drop-in could
load all 48 shards and then fail before opening the API:

```text
AttributeError: 'ShmRingBuffer' object has no attribute 'shared_memory'
```

The later TCPStore and NCCL broken-pipe messages on the peer rank were secondary.

### Root cause

The executor creates its scheduler-output `MessageQueue` before spawning `WorkerProc`, but the
worker normally opens the local reader only after `init_device()` and `load_model()`. During
that multi-minute gap, another process lifecycle can unlink the queue's POSIX SHM name. The
existing mapping remains valid in the creator, but a late reader can no longer open the name.

`ShmRingBuffer` also suppressed that `FileNotFoundError`, assuming the object had been
deserialized on another node. That left a half-constructed object and hid the useful failure
until its first dequeue (issue #26).

### Fix

- `WorkerProc.__init__` pre-opens only readers whose rank is local
  (`MessageQueue.preopen_from_handle`), before worker initialization or model loading.
- `MessageQueue` caches that live mapping by `(rank, buffer_name)`; the normal
  post-initialization `create_from_handle()` call consumes and reuses it.
- Remote readers are unchanged; they still initialize through the distributed process group
  after `init_device()`.
- A missing local segment now raises immediately with its SHM name and reader rank.
- Both files are `COPY`ed from `recipe/overlay/` into the overlay image, so the fix reaches
  every Stage-C node without a head-only bind mount.

### Safety

Local-only. Queues whose readers are remote return `None` from the pre-open step and keep their
original lifecycle, so nothing about cross-node messaging changes.

### Verification

The overlay image build includes a **CPU-only lifetime regression**
(`recipe/Dockerfile.dspark-runtime-overlay`): attach the local reader, unlink the POSIX name,
verify the normal late-open path reuses the live mapping, then verify a second uncached open
fails with the name/rank diagnostic. In a running container:

```bash
docker exec <container> grep -c "def preopen_from_handle" \
  "$VLLM_ROOT/distributed/device_communicators/shm_broadcast.py"   # >= 1
```

Before upstreaming, issue #26's reporter validated the same patch on the affected two-node
deployment: correct 1M model metadata, a real completion, six concurrent completions, and no
CUDA, NCCL, EngineDead, or request errors. That live system was not reused for PR testing;
maintainers should retain their requested two-node reproduction gate before merge.

### Configuration

None. Delivered baked-in — see [Delivery mechanisms](#delivery-mechanisms).

## Patch 6 (upstream #54) — prefix cache lost on long conversations

Adopted from
[PR #54](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/54).
Diff: [`patches/0006-kv-cache-prompt-protection-cap-and-swa-recycle.patch`](../patches/0006-kv-cache-prompt-protection-cap-and-swa-recycle.patch).
Source: `recipe/overlay/vllm/v1/core/single_type_kv_cache_manager.py`.

### Symptom

Long-context agent sessions see the first call of most turns re-prefill the whole prompt
(235 s for ~400K tokens) although identical resends hit 100%. For an agent, the first call of
a user turn cannot hit at the previous prompt boundary once the prior round's reasoning is
stripped — and any large intervening request (e.g. a background review) destroys the
conversation's prefix.

### Root cause

Two causes, both in `vllm/v1/core/single_type_kv_cache_manager.py`:

1. **Unbounded prompt-block protection.** `MLAAttentionManager.cache_blocks` (opted in for
   `model_version == "deepseek_v4"`) and `SlidingWindowMLAManager.cache_blocks` call
   `_protect_prompt_blocks`, which `touch()`es prompt blocks — an extra `ref_cnt` that survives
   the request. The only cap, `2 * max_model_len / block_size`, is 8,192–524,288 pages per
   group at `max_model_len=1M`, larger than the pool (~12,621 pages). The protected set grows
   by ~30–60 pages on *every* request, never shrinks while idle, and is released only FIFO
   under allocation pressure — evicting the oldest conversation's prefix first.
2. **Sliding-window churn through the shared LRU.** DSV4's compressor/indexer groups use pages
   of a few tokens. `remove_skipped_blocks` frees each skipped page to the global free queue and
   the next `allocate_new_blocks` pops fresh pages from the queue *head*. One 354K-token prefill
   cycles the whole pool several times, evicting every other request's cached pages — with any
   amount of free space. The protection in (1) was the fork's shield against (2).

### Fix

- **Cap** the protected set to a fraction of the pool, shared across the KV groups
  (`VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION`, default `0.30`; managers count themselves).
- **Recycle**: sliding-window groups keep the pages they skip in a per-request recycle list
  (`ref_cnt` kept, hash evicted) and reuse them for that request's next allocations; leftovers
  are freed at `free()`. Shared or protected pages (`ref_cnt != 1`) are freed as before.
  Admission accounting is unchanged (conservative: it ignores recyclable pages).

### Verification

Measured on 2× DGX Spark, TP=2, `nvfp4_ds_mla`, DeepSeek-V4-Flash-Vision-Exp, DSpark on.
"cap only" = protection cap; "cap + recycle" = the shipped pair:

| Scenario | stock build | cap only | cap + recycle |
|---|---|---|---|
| 354K FULL, resend | 100% · 1.3 s | 100% · 1.3 s | 100% · 1.2 s |
| 354K ALT, then FULL again | 0% · 236 s | 0% · 229 s | 100% · 1.2 s |
| ALT again | 0% · 235 s | 0% · 230 s | 100% · 1.3 s |
| Third tree, then FULL / ALT | — | — | 100% / 100% |
| FULL + 300-token decode, then resend | — | — | 100% (6.8 s decode) |
| Idle growth over 12 small requests | +0.47 pp each | +0.46 pp each | +0.48 pp each (capped later) |
| Idle usage after 80 small requests | ~37% (extrapolated) | 15.2% | stable (capped) |
| Greedy output vs stock | reference | identical | identical |
| 1200-token generation (thinking off) | — | — | coherent, 35.6 tok/s |
| `reasoning_effort=max` | — | — | reasoning + answer returned |

Small requests still pin pages until the cap engages (by design of the fork's protection); with
the cap the pool stabilises (15% after 80 requests here) instead of growing forever.

In-container presence check:

```bash
docker exec <container> grep -c VLLM_SWA_RECYCLE_SKIPPED_BLOCKS \
  "$VLLM_ROOT/v1/core/single_type_kv_cache_manager.py"    # >= 1
# The first recycling request also logs:
docker logs <container> 2>&1 | grep "recycling skipped pages in-request"
```

### Configuration

| Knob | Default | Effect |
|---|---|---|
| `VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION` (launcher: `PROTECTED_FRACTION`) | `0.30` | Share of the KV pool the DSv4 prompt-block protection may pin, summed over KV groups. `0` disables protection; `-1` restores the old unbounded behaviour. Non-numeric falls back to `0.30`. |
| `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS` (launcher: `SWA_RECYCLE`) | `1` | `0` disables in-request recycling of skipped sliding-window pages. |

**File names and launchers** (the originating analysis doc's staging names, mapped to this repo):

- The real unified diff is
  [`patches/0006-kv-cache-prompt-protection-cap-and-swa-recycle.patch`](../patches/0006-kv-cache-prompt-protection-cap-and-swa-recycle.patch).
  There is no `patch6-single_type_kv_cache_manager.diff` here — the file that stages at `/var/tmp`
  **is** `patch6-single_type_kv_cache_manager.py`, which is where that `.diff` name comes from.
- There is no launcher named `ds4-vision-tp2-spark-recycle.sh`. Both current launchers —
  [`../launchers/ds4-vision-tp2.sh`](../launchers/ds4-vision-tp2.sh) and
  [`../launchers/ds4-vision-tp4.sh`](../launchers/ds4-vision-tp4.sh) — carry the two knobs
  (`VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION` default `0.30`, `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS`
  default `1`).

## Patch A (optional) — drafter-private cudagraph capture sizes

**Optional and off by default**, and **not mounted by the current launchers**. With
`VLLM_DSPARK_DRAFT_CAPTURE_SIZES` unset the proposer behaves exactly as stock. Source:
[`patches/A-drafter-sizes/v1/spec_decode/dspark_proposer.py`](../patches/A-drafter-sizes/v1/spec_decode/dspark_proposer.py)
— the repo's overlay proposer plus this one feature, so mounting it does not drop Patches 1/2/2b.

### Symptom

The DSpark proposer shares the target model's cudagraph capture sizes. Under speculative
decoding those sizes round to multiples of `1+k` (k=5 → 6, 12, 18, …), so a batch-1 draft call
dispatches on the 6-bucket, is clipped to the 4 draft rows (`max_num_seqs`), and the draft MoE
processes **20 draft tokens per step for a single stream instead of 5**.

### Fix

`VLLM_DSPARK_DRAFT_CAPTURE_SIZES=1` installs a drafter-private capture-size view — powers of
two up to `max_num_seqs` (`{1, 2, 4}` at `max_num_seqs=4`) — captured in `dummy_run`. The
view is installed on the drafter's `CudagraphDispatcher` only, never on the shared
`VllmConfig`, so the target's uniform-decode FULL graphs keep their `1+k`-rounded sizes.
Default OFF is byte-identical to stock behaviour; a comma-separated list sets explicit
request-count buckets instead of the power-of-two default.

### Verification

**+3.0%** on the single-stream decode battery — chat +6.4%, code +6.5% — with quality probes
byte-identical. The battery's counting prompts are draft-acceptance ceilings rather than
throughput; see [`BENCHMARKS.md`](BENCHMARKS.md) for how to read the figures.

### Cost

~2.5 GB at profiling for the extra drafter graphs. At 909K context, run with
`--gpu-memory-utilization 0.78` to absorb it.

### Apply

Bind-mount `v1/spec_decode/dspark_proposer.py` over
`/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark_proposer.py:ro` on **both**
nodes, with `VLLM_DSPARK_DRAFT_CAPTURE_SIZES` set identically on both ranks — as with every
DSpark switch, a rank mismatch diverges the collective sequence and **NCCL hangs**. Confirm on
both ranks via the boot line `DSpark drafter-private cudagraph capture sizes enabled: [...]`.

### Status in this repo

Opt-in only: neither launcher mounts it and neither sets the variable, so the fleet runs the
stock shared-size behaviour until an operator opts in.

## Vision port — `ds4v_model` / `ds4v_vision` / `ds4v_mm` / `ds4v_registry`

Four bind-mounted files give the checkpoint native image input. Details:
[`../vision-exp/README.md`](../vision-exp/README.md).

### Symptom

Serving the checkpoint with the stock runtime fails two ways:

- `... no module or parameter named 'aligner' in DeepseekV4ForCausalLM` — the weight loader
  finds vision weights the class does not declare.
- `... is not a multimodal model` — even with the class extended, vLLM never treats it as one.

### Root cause

`DeepseekV4ForCausalLM` has no vision tower or aligner, so the checkpoint's `vision.*` /
`aligner.*` weights have nowhere to go. Separately, vLLM decides `is_multimodal_model` from a
**static architecture-name table** in the model registry, not from the class, so a class that
gains vision support is still classified as text-only.

### Fix

| staged file | container target | what it is |
|---|---|---|
| `/var/tmp/ds4v_model.py` | `models/deepseek_v4/nvidia/model.py` | the image's own `model.py` + `vision-exp/port/patch_vision.py` (ViT + aligner + mapper + `e_score_correction_bias_vl` gate + `vision.`/`aligner.` loader guard) |
| `/var/tmp/ds4v_vision.py` | `models/deepseek_v4/nvidia/ds4v_vision.py` | the ported ViT + Aligner (shipped in this repo) |
| `/var/tmp/ds4v_mm.py` | `models/deepseek_v4/nvidia/ds4v_mm.py` | multimodal processing / dummy inputs / processor (shipped in this repo) |
| `/var/tmp/ds4v_registry.py` | `model_executor/models/registry.py` | the image's own `registry.py` + `patch_registry.py`, adding the `DeepseekV4VForConditionalGeneration` multimodal alias selected by `--hf-overrides` |

`ds4v_model.py` and `ds4v_registry.py` are **derived from the image you actually run** rather
than checked in: both are copies of vLLM files, so a checked-in copy would silently pin you to
one image build and drift the moment the image moves. Generating them means the port always
applies to the running image, and the patchers fail loudly (anchor count ≠ 1) if the image
changed underneath them (issue #46).

### Verification

`build-ds4v-files.sh` verifies before installing — both files parse and carry every marker:

```text
verified: both files parse and carry every port marker
```

In a running container:

```bash
docker exec <container> grep -c DeepseekV4VForConditionalGeneration \
  "$VLLM_ROOT/model_executor/models/registry.py"          # >= 1
docker exec <container> grep -c "ds4v_vision" \
  "$VLLM_ROOT/models/deepseek_v4/nvidia/model.py"         # >= 1
```

**Clear the inspection cache after changing these files.** vLLM caches model inspection on disk
keyed by module + class; without clearing it, the alias reuses the stale pre-patch "text-only"
verdict:

```bash
rm -rf "$VLLM_CACHE_ROOT/modelinfos/"
```

### Configuration

| Item | Value |
|---|---|
| `--hf-overrides` | `{"architectures":["DeepseekV4VForConditionalGeneration"]}` (both launchers) — selects the registry alias |
| `vision-exp/build-ds4v-files.sh [IMAGE] [DEST]` | `$1` / `DSPARK_VLLM_IMAGE` = target image; `$2` = staging dir (default `/var/tmp`) |

## Delivery mechanisms

There are two routes to the container filesystem, and they carry the same code.

### 1. Baked into the runtime image

`recipe/overlay/` holds the overlaid vLLM sources;
[`../recipe/Dockerfile.dspark-runtime-overlay`](../recipe/Dockerfile.dspark-runtime-overlay)
`COPY`s them over the site-packages tree, and
[`../build-dspark-vllm-runtime.sh`](../build-dspark-vllm-runtime.sh) drives the build (overlay
stage, then the `nvfp4` stage A → B → C chain, on head and worker). The overlay build runs
three gates:

| Gate | What it catches |
|---|---|
| `py_compile` on every overlaid file | syntax errors |
| Import check (module imports **and** a named entry point resolves — e.g. `shm_broadcast.MessageQueue`, `multiproc_executor.WorkerProc`, `DeepSeekV4ToolParser` still subclasses `DeepSeekV32ToolParser`) | files that compile but die at import — the expensive failure mode, because `py_compile` alone yields a green build and a server that dies only after the weights load |
| CPU-only SHM lifetime regression (Patch 6 / upstream #30) | the queue-lifetime behaviour itself, with no GPU required |

### 2. Read-only bind mounts at `/var/tmp`

The vision launchers overlay three patched files plus the four vision files with `:ro` bind
mounts because the deployed image predates the baked patches, and because
`single_type_kv_cache_manager.py` is not `COPY`ed by the overlay Dockerfile at all (it reaches
the container only as a mount). Same code, different transport:

```text
/var/tmp/patch3-scheduler.py                  -> vllm/v1/core/sched/scheduler.py
/var/tmp/spec-dspark.py                        -> vllm/v1/spec_decode/dspark.py
/var/tmp/patch6-single_type_kv_cache_manager.py -> vllm/v1/core/single_type_kv_cache_manager.py
/var/tmp/ds4v_model.py                         -> vllm/models/deepseek_v4/nvidia/model.py
/var/tmp/ds4v_vision.py                        -> vllm/models/deepseek_v4/nvidia/ds4v_vision.py
/var/tmp/ds4v_mm.py                            -> vllm/models/deepseek_v4/nvidia/ds4v_mm.py
/var/tmp/ds4v_registry.py                      -> vllm/model_executor/models/registry.py
```

### Match the image or crash

**A proposer (or any overlaid module) mounted from a mismatched vLLM build crashes the runtime.**
`propose()` signatures, internal imports and symbol names move between vLLM versions; copying a
file from one build onto another produces errors like
`propose() got an unexpected keyword argument 'req_ids'`. This is why:

- the vision files are generated from the running image rather than checked in (issue #46);
- `docker-compose.dspark.yml` deliberately does **not** bind-mount a proposer copy ("a copy
  from one vLLM version crashes another");
- for third-party images, the guard should be applied in place with
  [`../scripts/apply-nonuniform-guard.py`](../scripts/apply-nonuniform-guard.py) instead of a
  copied file;
- every bind mount must target the path printed by the check scripts — a mount onto a wrong path
  is silently a no-op for the code that actually loads.

To confirm the Dockerfile's `COPY` list still matches the tree:

```bash
./scripts/verify-overlay-sources.sh [Dockerfile] [overlay-dir]
# exit 1 + "Missing overlay source referenced by ...: <path>" for any COPY with no source
```

## Staging and verification

Stage on **every** node before launch — the launchers check for the files but never copy them.

```bash
# 1. Vision-port files (run on every node, against the image you will run)
./vision-exp/build-ds4v-files.sh              # -> /var/tmp/ds4v_{model,vision,mm,registry}.py

# 2. Patched runtime files (run on every node)
cp recipe/overlay/vllm/v1/core/sched/scheduler.py               /var/tmp/patch3-scheduler.py
cp recipe/overlay/vllm/v1/spec_decode/dspark.py                 /var/tmp/spec-dspark.py
cp recipe/overlay/vllm/v1/core/single_type_kv_cache_manager.py  /var/tmp/patch6-single_type_kv_cache_manager.py
```

### Launcher preflight and exit codes

| Exit | Meaning | Where |
|---|---|---|
| `1` | Container exited immediately after `docker run` (see `docker logs`) | end of both launchers |
| `2` | Usage error — rank not in `0..1` (TP2) / `0..3` (TP4) | argument parsing |
| `3` | `MODEL MISSING` — `$MODELS_HOST/$MODEL_DIR` not a directory | model preflight |
| `4` | A required staged file is missing under `/var/tmp` | patch preflight |

- `launchers/ds4-vision-tp2.sh` checks `patch3-scheduler.py`, `spec-dspark.py` and
  `patch6-single_type_kv_cache_manager.py`; the missing-file message names the source path under
  `recipe/overlay/`. The vision files are **not** preflighted here, so a missing one fails at
  container start instead.
- `launchers/ds4-vision-tp4.sh` checks **all seven** staged files in a loop (`MISSING
  /var/tmp/<f>` → exit `4`).

### Post-boot checks (both nodes)

```bash
# Patch 3 — cold-prefill garble guard (is_prefill_chunk present)
./scripts/check-patch3.sh <head-container> <worker-container>

# Patch 4 — shared-expert mapping present (grep -c shared_experts >= 6)
./scripts/check-patch4.sh <head-container> <worker-container>
```

Both scripts: `exit 0` = present in every container, `exit 1` = missing somewhere,
`exit 2` = usage error. Both print `OK <container>` / `FAIL <container>` per container and the
resolved vLLM root.

**The vLLM package root is not the same on every image.** Hardcoding
`/opt/env/lib/python3.12/site-packages` makes a check report "No such file" — a FAIL that looks
like a missing patch — on a perfectly good deployment with a Debian layout
(`/usr/local/lib/python3.12/dist-packages`), as reported by **@robotnurse** in issue #22. The
checks ask Python where vLLM lives and fall back to probing known paths. Override explicitly with:

```bash
VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm ./scripts/check-patch4.sh <container>
```

Use that same resolved root for the manual greps in each patch section.

## Environment knobs

| Knob | Default | Set by launchers | Patch | Effect |
|---|---|---|---|---|
| `VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION` (`PROTECTED_FRACTION`) | `0.30` | both, `${PROTECTED_FRACTION:-0.30}` | 6 (upstream #54) | Share of the KV pool the DSv4 prompt-block protection may pin across all KV groups; `0` disables protection, `-1` restores the old unbounded behaviour |
| `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS` (`SWA_RECYCLE`) | `1` | both, `${SWA_RECYCLE:-1}` | 6 (upstream #54) | `0` disables in-request recycling of skipped sliding-window/compressor pages |
| `VLLM_SUPPRESS_STOPS_IN_REASONING` | on (`"1"`) | not set | 5 (optional) | `0` disables the stop-in-reasoning guard, process-wide |
| `VLLM_DSPARK_DRAFT_CAPTURE_SIZES` | off (unset) | not set | A (optional) | `1`/`auto` = drafter-private capture sizes (powers of two up to `max_num_seqs`); a comma-separated list sets explicit request-count buckets (values above `max_num_seqs` dropped, `max_num_seqs` always appended); `0`/`off` = share the target's `1+k`-rounded sizes |
| `DSPARK_SLOT_CLAMP` | `1` | both, `=1` | 1 / slot safety | `0` reverts the protective out-of-range `slot_index` clamp to detect-and-log only (A/B kill switch) |
| `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK` | — | both, `=1` | 2 / 2b | Selects the only path made ragged; without `1` the rectangular path runs and Patch 2/2b do not protect mixed batches |
| `VLLM_ROOT` | autodetected | not set | verification only | Overrides the container-side vLLM package root for `check-patch3.sh` / `check-patch4.sh` |
| `DSPARK_VLLM_IMAGE` | build-script default; `build-ds4v-files.sh` falls back to `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b` | build script | vision port | Target image for `vision-exp/build-ds4v-files.sh` (overridable as `$1`) |

`VLLM_SWA_RECYCLE_SKIPPED_BLOCKS` is read once at module import, so changing it takes effect
on the next container start. `VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION` is read at call time, but
the effective cap is logged the first time each KV group consults it
(`protected prompt block cap for KV group N = ...`), which is where you confirm what the fleet
is actually running.
