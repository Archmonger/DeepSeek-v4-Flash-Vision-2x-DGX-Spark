# DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark — native vision, TP2, DSpark

This directory adds native multimodal support for
[`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)
to the DSpark vLLM runtime. Not a sidecar, not a VLM proxy — the model's own ViT and
aligner running inside vLLM, with DSpark speculative decoding intact.

```
$ curl :8888/v1/chat/completions -d @image_request.json

  "Based on the image, the two colors are **red** and **blue**.
   *   **Red** is on the **left** side.
   *   **Blue** is on the **right** side."
```

The runnable recipe and current numbers are in [`CURRENT.md`](../CURRENT.md); the long-form flag
explanation is in [`VISION-EXP-DEFAULT-CONFIG.md`](../VISION-EXP-DEFAULT-CONFIG.md).

## Why this needed a port at all

vLLM's `DeepseekV4ForCausalLM` is the **text-only** class. The Vision-Exp checkpoint
reports the *same* `architectures` string but carries 316 extra tensors vLLM has no home
for, so loading it dies immediately:

```
ValueError: There is no module or parameter named 'aligner' in DeepseekV4ForCausalLM
```

DeepSeek shipped only a reference implementation — its own `inference/README.md` says it is
"a readable reference implementation rather than a production serving engine." There was no
vLLM path to configure.

## What the Vision-Exp config carries

Four groups of fields in this checkpoint's `config.json` have no counterpart in the text-only
vLLM class:

| field | this checkpoint |
|---|---|
| vision keys | 10 (`vision_n_layers` 32, `vision_dim` 1024, patch size 14, 2D RoPE, downsample 3) |
| `num_nextn_predict_layers` | **3** — the DSpark drafter is 3 layers |
| `rms_norm_eps` | 1e-20 |
| `ffn.gate.*` | `weight`, `tid2eid`, **`bias`**, **`bias_vl`** |

The last row is the interesting one: a **modality-specific MoE routing bias**. `bias`
applies to text tokens, `bias_vl` to image tokens — experts are chosen differently by
modality. It appears on **all 43 layers**, including the 3 hash-routing layers. No paper or
model card documents it.

## Staging the four bind-mounted files

[`launchers/ds4-vision-tp2.sh`](../launchers/ds4-vision-tp2.sh) bind-mounts four files from
`/var/tmp`. **Generate them with one command, on every node:**

```bash
git clone https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark
cd DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/vision-exp
./build-ds4v-files.sh                      # or: ./build-ds4v-files.sh <image-tag> <dest>
```

Two of the four are shipped in this repo and copied verbatim (`ds4v_vision.py`,
`ds4v_mm.py`). The other two are **derived from the image you are actually running**:

| file | provenance |
|---|---|
| `ds4v_vision.py` | shipped — `vision-exp/port/ds4v_vision.py` |
| `ds4v_mm.py` | shipped — `vision-exp/port/ds4v_mm.py` |
| `ds4v_model.py` | *generated* — image's `deepseek_v4/nvidia/model.py` + `port/patch_vision.py` |
| `ds4v_registry.py` | *generated* — image's `model_executor/models/registry.py` + `port/patch_registry.py` |

**Why the last two are generated rather than checked in:** both are copies of vLLM files.
A checked-in copy would pin you to one image build and drift silently the moment the image
moves. Generating them means the port always applies to the image you run, and the
patchers fail loudly (anchor count != 1) if the image changed underneath them.

The script verifies its own output before installing — both files must parse, and
`ds4v_model.py` must carry the vision import, the `bias_vl` gate and the loader guard,
while `ds4v_registry.py` must carry the multimodal alias. A patcher that silently no-ops
fails here instead of surfacing later as `is not a multimodal model`.

Patch 4 (`spec-dspark.py`) is a **separate, required** mount and is not part of the four
vision files. Without it the draft's always-on shared expert loads uninitialised and decode
runs at roughly half speed, silently — see
[`DSPARK-SHARED-EXPERT-FIX.md`](../DSPARK-SHARED-EXPERT-FIX.md).

## The port

| file | what it does |
|---|---|
| `port/ds4v_vision.py` | The ViT (32 blocks, 2D RoPE, RMSNorm, gated MLP) + Aligner (pixel-shuffle ÷3 → 2-layer GELU MLP). Verbatim numerics from the checkpoint's `inference/vision.py`. Deliberately **not** TP-sharded — ~410M params, cheaper to replicate than to all-gather. |
| `port/ds4v_mm.py` | vLLM multimodal plumbing: processing info, dummy inputs, and a custom processor. The checkpoint has no HF processor, so preprocessing (resize solver, patchify, N-layout block build) is ported from `inference/image_processor.py`. |
| `port/patch_vision.py` | Idempotent patcher for vLLM's vendored `deepseek_v4/nvidia/model.py` — 11 anchored edits. |
| `port/patch_registry.py` | Registers a multimodal architecture alias (see below). |
| [`../launchers/ds4-vision-tp2.sh`](../launchers/ds4-vision-tp2.sh) | TP2 launcher (canonical path is `launchers/`; `vision-exp/ds4-vision-tp2.sh` is a symlink kept for older PR/issue links). The validated TP2 command from [`VISION-EXP-DEFAULT-CONFIG.md`](../VISION-EXP-DEFAULT-CONFIG.md) plus the vision mounts and `--hf-overrides`. |
| [`../launchers/ds4-vision-tp4.sh`](../launchers/ds4-vision-tp4.sh) | TP4 launcher, all four Sparks. Same file extended to 4 nodes, at `max-num-seqs 64` / `max-cudagraph-capture-size 64`. |

Every patch is guarded on `vision_n_layers > 0`, so with no vision layers in the config every
guarded branch resolves to the stock vLLM path and a text-only run through these files is left
untouched.

## Twelve things that had to be fixed

Each was a real error, in the order they surfaced:

1. **`no module or parameter named 'aligner'`** — registered ViT + Aligner + the four learned
   embeddings (`image_start/end/newline/pad`) and taught the weights mapper their prefixes.
2. **`KeyError: aligner.gate_up_proj.bias`** — vLLM's fused-MLP `stacked_params_mapping`
   rewrites any `.w1`/`.w3` into `gate_up_proj`. The ViT MLP and the aligner legitimately use
   `w1`/`w2`. Guard so they fall through to the generic loader.
3. **`KeyError: layers.0.ffn.gate.e_score_correction_bias`** — the per-layer gate bias,
   including on the hash-MoE layers vLLM explicitly skips ("hash MoE doesn't use
   e_score_correction_bias" — not true of this checkpoint, which carries one there too). Plus
   `bias_vl`.
4. **`'DeepseekV4Config' object has no attribute 'image_token_index'`** — the DSpark proposer
   expects the standard VLM field once the model reports multimodal. Published from the
   tokenizer's `<｜deepseek_image｜>` id (129264).
5. **`Target model does not have 'model' attribute`** — the proposer calls
   `target_model.get_language_model()` then `.model`/`.lm_head`. Standard VLMs keep the tower
   beside a separate `language_model`; here the ViT lives *inside* the decoder stack, so
   `get_language_model()` returns `self`.
6. **`is not a multimodal model`** — vLLM answers `is_multimodal_model` from a **static
   architecture-name table**, never inspecting the class. `SupportsMultiModal` in the MRO is
   not enough. Added a `DeepseekV4VForConditionalGeneration` alias in `_MULTIMODAL_MODELS`
   pointing at the same class, selected via `--hf-overrides`; a config with no `vision_n_layers`
   keeps the plain text entry.
7. **Same error, still** — vLLM caches model inspection on disk in
   `$VLLM_CACHE_ROOT/modelinfos/`, keyed by **module + class**. Both names resolve to the same
   class, so the alias reused the stale pre-patch "text-only" entry. Clear `modelinfos/`
   after changing model interfaces.
8. **`IndexError: list index out of range` in `_merge_mm_kwargs`** — ragged per-image fields
   must use `MultiModalFieldConfig.flat_from_sizes`, not `batched`. Same shape as
   `deepseek_vl2.py`.
9. **Processor received 0 images** — vLLM passes `mm_data['images']` (plural, HF convention);
   reading `'image'` silently yielded nothing.
10. **`0 prompt placeholders`** — vLLM sets `is_update_applied=True` on the text+mm path and
    only *searches* the returned ids, so the placeholder must be expanded to the full block
    **inside `_call_hf_processor`**. It is also the only place the position-dependent
    `COMPRESS_PAD_TO` alignment can be expressed, since a replacement callable only gets
    `item_idx`.
11. **`mat1 and mat2 shapes cannot be multiplied (3x196 and 588x1024)`** — `flat_from_sizes`
    hands back one concatenated tensor, not a per-image list; iterating it walked individual
    14x14 patches into the patch embedder. Split on `num_patches`.
12. **`DeepSeek V4 hash MoE routing requires input_ids`** — the first `num_hash_layers` MoE
    layers route by **token id**, but vLLM's multimodal path passes `inputs_embeds` with
    `input_ids=None`. Fixed with `requires_raw_input_tokens = True`, which keeps the raw ids
    alongside the embeddings.

## Measured (2x DGX Spark GB10, TP2, temperature 0)

Vision-side measurements with the port in place:

| | |
|---|---|
| **Vision, 112x112 image** | correct on colour *and* side, both orientations |
| **Vision, 336x336 + 26-token answer** | 1.03 s end to end |
| **Image block size** | 112x112 → 117 tokens · 168x168 → 143 · 336x336 → 129 |

Image-block token counts match the reference math. For the validated profile's context, KV pool and
decode numbers see [`CURRENT.md`](../CURRENT.md) and
[`VISION-EXP-DEFAULT-CONFIG.md`](../VISION-EXP-DEFAULT-CONFIG.md). KV pool size is a **per-boot**
figure rather than a property of the config: two boots of an identical config can differ by ~11%,
because available KV memory on GB10 varies with what else has touched unified memory
([`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md)).

### Speculative depth: `k = 5`

**`MTP_NUM_TOKENS=5` is the validated setting** — what the main recipe uses and what
[`launchers/ds4-vision-tp2.sh`](../launchers/ds4-vision-tp2.sh) and
[`launchers/ds4-vision-tp4.sh`](../launchers/ds4-vision-tp4.sh) pass. Measured **with Patch 4**
mounted on a second 2× DGX Spark ([issue #48](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/issues/48);
warm, temp 0, 500K context):

| workload | k=5 | k=3 |
|---|---:|---:|
| count to 300 | **85.5 tok/s** (accept 0.974 · 5.88 tok/step) | 64.4 tok/s |
| code | 49.9 | 49.5 |
| prose | 29.5 | 32.1 |

Counting numbers are a draft-acceptance ceiling, not throughput — see "How we quote numbers" in
[`CURRENT.md`](../CURRENT.md). `num_nextn_predict_layers: 3` does not imply `k=3`: any `k=3`
comparison run **without** the Patch 4 mount measures the loader rather than the drafter, because
an uninitialised shared expert collapses acceptance to the loader's own signature
([`DSPARK-SHARED-EXPERT-FIX.md`](../DSPARK-SHARED-EXPERT-FIX.md)). The launcher in this directory
mounts Patch 4 ([PR #44](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/pull/44)).
With the patch in place, `k=5` wins clearly on predictable work, and `k=3` keeps a small edge on
prose.

### The limiter here is per-step cost, not speculation

Acceptance is high on predictable work, so the headroom left on this build sits in decode step cost
rather than in the drafter. Two candidates:

1. **Vision tax.** This build sets `requires_raw_input_tokens = True` so the runner slices
   and passes raw token ids every forward step (the text-only path does not), and the ViT
   plus aligner stay resident on both ranks.
2. **Context ceiling.** A larger `max_model_len` means larger DSA indexer buffers per step.

A single reload at a smaller context with everything else held constant separates the two.

### Correctness spot-checks (temperature 0)

- red-left / blue-right, 112x112 → *"Red is on the left side. Blue is on the right side."*
- green-top / yellow-bottom, 168x168 → *"Green and Yellow. The split is: Horizontal. The color on top is: Green."*

Different colours, different split axis, different image sizes — all correct, and the
image-block token counts match the reference math.

## Known deviations from the reference — read before claiming parity

Two, both quality-affecting and neither crash-causing. They are why this ships as a working
deployment rather than a parity claim.

**1. Bidirectional attention within image spans is not implemented.** The reference computes
`get_image_visible()` and widens the sparse-attention window so tokens inside an
`[IMAGE_START, IMAGE_END]` span attend bidirectionally. Here image tokens use the standard
causal sparse pattern, so a token sees at most 128 of a span up to 384 tokens, and no later
patches at all. The closest measured analogue (vLLM
[#40106](https://github.com/vllm-project/vllm/issues/40106), Gemma-4) shows KL 0.03–0.09
concentrated at image positions. Expect degradation on dense-intra-image work — OCR, charts,
documents — and note it will still pass a smoke test.

**2. `bias_vl` is loaded but not applied.** Image tokens currently route through the text
gate bias. Because every image slot carries the same placeholder id, hash routing also sends
them all to one expert — which is very likely the reason `bias_vl` exists. Correct handling
needs modality threaded into the MoE gate.

Neither affects text-only requests: with no image tokens the code path is the stock vLLM one.

## Run it

```bash
# both nodes: extract vLLM's model.py from the image, patch, and stage
docker run --rm -v /var/tmp:/out --entrypoint bash $IMAGE -lc \
  'cp /opt/env/lib/python3.12/site-packages/vllm/models/deepseek_v4/nvidia/model.py /out/ds4v_model.py
   cp /opt/env/lib/python3.12/site-packages/vllm/model_executor/models/registry.py /out/ds4v_registry.py'
sudo chown $(id -u):$(id -g) /var/tmp/ds4v_model.py /var/tmp/ds4v_registry.py
python3 port/patch_vision.py   /var/tmp/ds4v_model.py
python3 port/patch_registry.py /var/tmp/ds4v_registry.py
cp port/ds4v_vision.py port/ds4v_mm.py /var/tmp/

# stale inspection cache MUST go after changing model interfaces
sudo rm -rf ~/.cache/vllm-dspark/modelinfos

# worker first, then head
../launchers/ds4-vision-tp2.sh 1     # bluey  (holds weights, NFS-exports them)
../launchers/ds4-vision-tp2.sh 0     # asusi  (head, serves :8888)
```

Everything from the base recipe is unchanged: `--kv-cache-dtype nvfp4_ds_mla`,
`--block-size 256`, `draft_sample_method: probabilistic`, patch 3, and the full NCCL/env
block. Use the validated agent-serving profile — **1M context (1,048,576), gmu 0.85, seqs 12,
k=5** — with the Patch 4 mount present on every node. See [`CURRENT.md`](../CURRENT.md).

> **gmu 0.85, not less.** Don't dial `--gpu-memory-utilization` down looking for headroom: 0.80
> "boots and passes smoke tests, then dies under traffic" (issue #8), because DSpark allocates
> buffers on the *first real request*. The validated agent profile is **0.85**, documented in
> [`VISION-EXP-DEFAULT-CONFIG.md`](../VISION-EXP-DEFAULT-CONFIG.md).

## Prior art

vLLM issue [#54561](https://github.com/vllm-project/vllm/issues/54561) and draft PR
[#54566](https://github.com/vllm-project/vllm/pull/54566) cover a parallel implementation
validated on 2x RTX PRO 6000. Their fixes for the hash-routing guard, OOV sentinels, and
`bias_vl` are worth reading — this port reaches the same conclusions independently on several
points. **What is new here is GB10 / DGX Spark**: the
[NVIDIA forum position](https://forums.developer.nvidia.com/t/deepseek-v4-flash-vision-exp-is-released-as-open-weights/381911)
is that the native vLLM vision processor does not work with this model, and the standing answer
for vision on Spark is a sidecar VLM rather than a port.
