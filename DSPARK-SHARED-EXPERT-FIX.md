# The DSpark shared-expert loader bug — the draft's always-on expert loads uninitialised, silently

**Applies to:** any vLLM serving `DeepSeek-V4-Flash-Vision-Exp` with `method: dspark` — i.e. this recipe
**Impact on this deployment:** decode runs at roughly half speed until the Patch 4 mount is in place.
Restoring it takes count-to-100 **50.7 → 80.1 tok/s**, code **47.2 → 51.8**, prose **30.4 → 33.2**
(2× DGX Spark, warmed, single-stream, temp 0).
**Fix:** two lines — [`patches/0004-dspark-shared-expert-gate-up-proj.patch`](patches/0004-dspark-shared-expert-gate-up-proj.patch)

---

## Symptom

Decode runs at roughly half the expected rate while **output quality is perfect** — no garble, no
drift, no special-token leakage. Only speed changes.

The give-away is in the spec-decode metrics, not the throughput number: **drafted throughput stays
healthy while accepted throughput collapses.** Drafting itself is intact, so the deficit is in what
the target accepts, not in how fast a step runs. That is an acceptance failure, not a step-time
failure.

## Why it looks like a model regression (and isn't)

`tok/s = steps/s × accepted-tokens-per-step`. Under this bug the step rate is unchanged — the
engine, the fabric, the KV cache and the target model are all healthy. The entire deficit lands in
`accepted-tokens-per-step`.

Because speculative decoding is **verified by the target model**, a bad draft can never corrupt
output — it can only cost speed. So a broken drafter presents as "these weights are slow", which is
exactly the wrong place to look.

## Root cause

The DSpark draft's FFN is a `DeepseekV4MoE` containing a shared expert built as a `DeepseekV4MLP`,
whose projections are `gate_up_proj` (a `MergedColumnParallelLinear`, fed by checkpoint tensors
`w1` and `w3`) and `down_proj` (fed by `w2`).

The draft weight loader renames only `w2`:

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

So `shared_experts.w1` and `shared_experts.w3` match nothing, fall through to:

```python
# vllm/models/deepseek_v4/nvidia/dspark.py:1122-1125
param = params_dict.get(name)
if param is None:
    logger.debug("Skipping unknown DSpark weight %s", name)
    continue
```

…and are dropped. `logger.debug` is invisible at the default INFO level, so **the load reports
success**.

**12 checkpoint tensors are lost** — `w1` and `w3`, each as `weight` + `weight_scale_inv` (renamed
from `.scale` before the mapping is consulted), across all three draft stages. That leaves these six
parameters uninitialised:

```
model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.weight
model.layers.{43,44,45}.ffn.shared_experts.gate_up_proj.weight_scale_inv
```

`n_shared_experts: 1`, and the shared expert is **always-on** — its output is summed into every
token, unconditionally, alongside the routed experts. So each of the three draft stages runs with
its always-active expert uninitialised. The drafter still produces fluent, plausible tokens; they
just disagree with the target far more often.

### The tell: the target loader has the rows the draft loader is missing

```python
# vllm/models/deepseek_v4/nvidia/model.py:1952-1953   (target — correct)
("gate_up_proj", "w1", 0),
("gate_up_proj", "w3", 1),
```

The draft loader lost them when its mapping was narrowed to avoid a name collision — the DSpark
checkpoint also contains `markov_head.markov_w1`, which must not be treated as an FFN `w1` shard.
The narrowing worked, but took the shared-expert shards with it.

## The fix

```python
# vllm/v1/spec_decode/dspark.py:15-18   (AFTER)
_STACKED_PARAM_NAME_MAPPING = (
    ("attn.fused_wqa_wkv", ".attn.wq_a", 0),
    ("attn.fused_wqa_wkv", ".attn.wkv", 1),
    ("shared_experts.gate_up_proj", ".shared_experts.w1", 0),
    ("shared_experts.gate_up_proj", ".shared_experts.w3", 1),
)
```

**Why this is safe:**

- `map_dspark_stacked_param_name()` returns early on `".experts."` — note the **leading dot** — so
  routed experts (`ffn.experts.0.w1`) are untouched and still go through `expert_mapping`.
- The new patterns are anchored on the full `".shared_experts.wN"` segment. `markov_w1` has no
  `.shared_experts.` prefix and cannot match. Verified explicitly:

  ```
  43.ffn.shared_experts.w1.weight            -> ('...shared_experts.gate_up_proj.weight', 0)
  43.ffn.shared_experts.w1.weight_scale_inv  -> ('...shared_experts.gate_up_proj.weight_scale_inv', 0)
  44.ffn.shared_experts.w3.weight            -> ('...shared_experts.gate_up_proj.weight', 1)
  43.ffn.experts.0.w1.weight                 -> None      (routed expert, unchanged)
  45.markov_head.markov_w1.weight            -> None      (no collision)
  43.attn.wkv.weight                         -> ('...attn.fused_wqa_wkv.weight', 1)
  ```

- `.scale` → `.weight_scale_inv` renaming happens at `dspark.py:1074-1080`, *before* the mapping is
  consulted, and `_EXPERT_SCALE_RE = r"\.experts\.\d+\.w[123]\.scale$"` requires a digit after
  `.experts.`, so `shared_experts` correctly takes the `.weight_scale_inv` branch that
  `MergedColumnParallelLinear` expects for FP8 block quantisation.
- A wrong mapping **fails loudly** — `dspark.py:1086-1087` raises `KeyError` rather than skipping.
  A clean boot is therefore positive evidence the mapping resolved.

## Effect on this deployment

Same hardware (2× DGX Spark, TP=2), same runtime, same checkpoint, same flags — the loader mount is
the only variable. Warmed, single-stream, temp 0:

| workload | stock loader | **patched** |
|---|---:|---:|
| count to 100 | 50.7 | **80.1** |
| code | 47.2 | **51.8** |
| prose | 30.4 | **33.2** |

The gain on *count* carries a warm-up component on top of the loader fix, and acceptance on
prose-heavy traffic stays around 25% on this vision variant, which is why code and prose gain less
than count. Patch 4 is a correctness fix regardless of how the per-workload speed is attributed:
the always-on shared expert was loading uninitialised, and now it is not. Profile-level numbers for
the whole recipe live in [`CURRENT.md`](CURRENT.md) and
[`VISION-EXP-DEFAULT-CONFIG.md`](VISION-EXP-DEFAULT-CONFIG.md); how to read them is in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## How to check whether you are hit

You are affected if you serve `DeepSeek-V4-Flash-Vision-Exp` — or any DeepSeek-V4-Flash model
under DSpark — on a vLLM build whose `_STACKED_PARAM_NAME_MAPPING` lacks the `shared_experts`
rows.

```bash
# 1. does your checkpoint carry shared-expert draft tensors? (expect 18: w1/w3 = 12, w2 = 6)
python3 - <<'PY'
import json, glob
wm = json.load(open(glob.glob("<MODEL_DIR>/*.index.json")[0]))["weight_map"]
hits = sorted(k for k in wm if "mtp" in k and "shared_experts" in k)
print(len(hits), "shared-expert mtp tensors")
print("\n".join(hits[:6]))
PY

# 2. does your runtime's mapping include them?
grep -A6 '_STACKED_PARAM_NAME_MAPPING' \
  /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py

# 3. definitive: DEBUG logging prints every dropped tensor at load
#    VLLM_LOGGING_LEVEL=DEBUG ... then:
docker logs <container> 2>&1 | grep "Skipping unknown DSpark weight"
```

Apply with a read-only bind mount — no rebuild required:

```bash
-v /var/tmp/spec-dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro
```

## This is the mount the vision port loses

The vision serving port (`DeepSeek-V4-Flash-Vision-Exp`, this recipe's default — see
[`VISION-EXP-DEFAULT-CONFIG.md`](VISION-EXP-DEFAULT-CONFIG.md)) carries several read-only bind
mounts: the Patch 3 scheduler file, the four `ds4v_*.py` vision-model files, and this
`spec-dspark.py` (Patch 4) file. When the run command is assembled without `spec-dspark.py`, the
stock loader takes over, the draft's always-on shared expert loads uninitialised, and decode runs at
**roughly half speed** with perfect output quality — so nothing looks broken. It fails **silently**:
the dropped tensors are reported at `logger.debug`, invisible at INFO, and the load "reports
success".

**Restore the one mount** and verify it landed on **every** node (head *and* worker):

```bash
-v /var/tmp/spec-dspark.py:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py:ro
```

```bash
# Expect 6 — the two shared-expert mapping rows plus their comment. Stock loader returns 0.
docker exec <container> grep -c shared_experts \
  /opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark.py     # -> 6

# Definitive: with VLLM_LOGGING_LEVEL=DEBUG, no "Skipping unknown DSpark weight" for shared_experts:
docker logs <container> 2>&1 | grep "Skipping unknown DSpark weight.*shared_experts"   # -> (empty)
```

Or run [`scripts/check-patch4.sh <head-container> <worker-container>`](scripts/check-patch4.sh),
which resolves the vLLM package root per image and fails closed.

## What this is not

Recorded so nobody repeats the search. Every item below was measured, not assumed:

| tried | result |
|---|---|
| `draft_sample_method` greedy vs probabilistic | no change — it is a **no-op** for DSpark; the proposer never populates draft probs unless `VLLM_DSPARK_EXPORT_DRAFT_PROBS=1` |
| `fp8_ds_mla` vs `nvfp4_ds_mla` KV | no change — KV dtype is a context lever, not an acceptance lever |
| temperature 0 vs 0.7 | no change |
| B12X kernels off (`VLLM_USE_B12X_MOE=0`, `_WO_PROJECTION=0`) | **worse** — the step rate drops, then the run crashes |
| dedicated node pair, zero competing traffic | no change — contention was never involved |
| runtime / image build differences | identical vLLM version and identical DSpark env across nodes |
| draft shared-expert tensor names and dtypes | correct — the tensors exist with the expected names, shapes and dtypes; they are simply never loaded |
| `VLLM_DSPARK_CONFIDENCE_THRESHOLD` / `_SCHEDULER` | inert — `0.0`/`off` means *no gating*, and the confidence head is skipped on the hot path |

## Speculative depth: `k ≤ 5`, or a multiple of 5

Two properties of this drafter are worth stating precisely, because both look like tunable knobs and
neither is:

1. **`k` above the draft block size crashes at generation time, not only at boot.** The boot-time
   guard (`num_speculative_tokens % n_predict != 0`) can be patched out — but the run then
   **crashes on the first generation**:

   ```
   RuntimeError: The size of tensor a (7) must match the size of tensor b (5)
   ```

   The drafter emits exactly `dspark_block_size` (5 on this image) tokens per pass and `propose()`
   calls it once; multi-block drafting is not implemented. Raising `k` past the block size on
   DeepSeek-V4-Flash needs real proposer work, not a flag.

2. **The divisibility rule is `k ≤ 5`, or a multiple of 5.** The guard only fires when
   `k > n_predict`, which is why `k=3` and `k=4` boot fine. Upstream 0.25.2 uses the **same** `>`
   check, but resolves `n_predict` to `num_nextn_predict_layers` rather than `dspark_block_size`, so
   the guard never fires there; and its DSpark speculator sizes the draft block from `k`, so no shape
   mismatch follows. See issue #22.

Where `k=7` *does* work is on drafters whose draft block is at least 7 — e.g. MiMo-V2.5 DFlash
(`block_size` = 8, run at `num_speculative_tokens: 7`), GLM-5.2's DSpark speculator (`block_size` =
8, and its own `speculators_config` asks for 7 speculative tokens), and the Inkling DSpark preview
(`dspark_block_size` = `n_predict` = 7). DeepSeek-V4-Flash's draft block is 5, so **`k=5`** is what
this recipe passes.
