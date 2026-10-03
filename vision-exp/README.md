# DeepSeek-V4-Flash-Vision-Exp on 2x DGX Spark — native vision, TP2, DSpark

Native multimodal support for [`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) in the DSpark vLLM runtime. Not a sidecar, not a VLM proxy — the model's own ViT and aligner run inside vLLM, with DSpark speculative decoding intact.

```
$ curl :8888/v1/chat/completions -d @image_request.json

  "Based on the image, the two colors are **red** and **blue**.
   *   **Red** is on the **left** side.
   *   **Blue** is on the **right** side."
```

Pins, node map and live numbers: [`../CURRENT.md`](../CURRENT.md). The TP2 `vllm serve` command explained flag by flag: [`../docs/LAUNCH-FLAGS.md`](../docs/LAUNCH-FLAGS.md). Every patch, with symptom, root cause, fix and verification: [`../docs/PATCHES.md`](../docs/PATCHES.md).

## Why this needed a port at all

vLLM's `DeepseekV4ForCausalLM` is the **text-only** class. The Vision-Exp checkpoint reports the *same* `architectures` string but carries 316 extra tensors vLLM has no home for, so loading it dies immediately:

```
ValueError: There is no module or parameter named 'aligner' in DeepseekV4ForCausalLM
```

DeepSeek shipped only a reference implementation — its own `inference/README.md` calls it "a readable reference implementation rather than a production serving engine." There is no vLLM path to configure.

## What the Vision-Exp config carries

Four groups of fields in this checkpoint's `config.json` have no counterpart in the text-only class:

| field | this checkpoint |
|---|---|
| vision keys | 10 (`vision_n_layers` 32, `vision_dim` 1024, patch size 14, 2D RoPE, downsample 3) |
| `num_nextn_predict_layers` | **3** — the DSpark drafter is 3 layers |
| `rms_norm_eps` | 1e-20 |
| `ffn.gate.*` | `weight`, `tid2eid`, **`bias`**, **`bias_vl`** |

The last row is a **modality-specific MoE routing bias**: `bias` applies to text tokens, `bias_vl` to image tokens, so experts are chosen differently by modality. It appears on **all 43 layers**, including the 3 hash-routing layers. No paper or model card documents it.

## Run it

Primary path is sparkrun — one command, nothing staged by hand:

```bash
sparkrun run ../sparkrun/ds4-vision-exp-tp2.yaml
```

The **legacy** shell path stages the payload on every node first. `build-ds4v-files.sh` extracts
vLLM's files from the image you point it at, patches them, verifies them, and installs them into
`/var/tmp`; then launch worker-first:

```bash
../scripts/build/build-ds4v-files.sh    # or: ../scripts/build/build-ds4v-files.sh <image-tag> <dest>
../scripts/launch/ds4-vision-tp2.sh 1   # worker (holds the weights, NFS-exports them)
../scripts/launch/ds4-vision-tp2.sh 0   # head, serves :8888
```

Operating that path — staging, exit codes, the TP4 variant: [`LEGACY-LAUNCHERS.md`](../docs/LEGACY-LAUNCHERS.md).

Two of the four bind-mounted files ship in this repo; the other two are **derived from the image you are actually running**:

| file | provenance |
|---|---|
| `ds4v_vision.py` | shipped — `port/ds4v_vision.py` |
| `ds4v_mm.py` | shipped — `port/ds4v_mm.py` |
| `ds4v_model.py` | *generated* — image's `deepseek_v4/nvidia/model.py` + `port/patch_vision.py` |
| `ds4v_registry.py` | *generated* — image's `model_executor/models/registry.py` + `port/patch_registry.py` |

**Why generated rather than checked in:** both are copies of vLLM files, so a checked-in copy pins you to one image build and drifts silently the moment the image moves. Generating them means the port always applies to the image you run, and the patchers fail loudly (anchor count != 1) if the image changed underneath them. The script checks its own output before installing — both files must parse, `ds4v_model.py` must carry the vision import, the `bias_vl` gate and the loader guard, and `ds4v_registry.py` must carry the multimodal alias — so a patcher that silently no-ops fails there instead of surfacing later as `is not a multimodal model`. Stale `$VLLM_CACHE_ROOT/modelinfos/` must still go after any model-interface change (fix 7 below).

**gmu 0.85, not less.** 0.80 boots and passes smoke tests, then dies under traffic, because DSpark allocates buffers on the *first real request*. The validated profile is 0.85.

## The port

| file | what it does |
|---|---|
| `port/ds4v_vision.py` | The ViT (32 blocks, 2D RoPE, RMSNorm, gated MLP) + Aligner (pixel-shuffle ÷3 → 2-layer GELU MLP). Verbatim numerics from the checkpoint's `inference/vision.py`. Deliberately **not** TP-sharded — ~410M params, cheaper to replicate than to all-gather. |
| `port/ds4v_mm.py` | vLLM multimodal plumbing: processing info, dummy inputs, a custom processor. The checkpoint ships no HF processor, so preprocessing (resize solver, patchify, N-layout block build) comes from `inference/image_processor.py`. |
| `port/patch_vision.py` | Idempotent patcher for vLLM's vendored `deepseek_v4/nvidia/model.py` — 11 anchored edits. |
| `port/patch_registry.py` | Registers a multimodal architecture alias. |
| [`../sparkrun/ds4-vision-exp-tp2.yaml`](../sparkrun/ds4-vision-exp-tp2.yaml) | **Primary TP2 recipe.** Same serve argv as the launcher, payload staged in-container — [`SPARKRUN-PARITY.md`](../docs/SPARKRUN-PARITY.md). |
| [`../scripts/launch/ds4-vision-tp2.sh`](../scripts/launch/ds4-vision-tp2.sh) | **Legacy** TP2 launcher (`vision-exp/ds4-vision-tp2.sh` is a symlink kept for older PR/issue links); flags explained in [`LAUNCH-FLAGS.md`](../docs/LAUNCH-FLAGS.md), how to run it in [`LEGACY-LAUNCHERS.md`](../docs/LEGACY-LAUNCHERS.md). |
| [`../scripts/launch/ds4-vision-tp4.sh`](../scripts/launch/ds4-vision-tp4.sh) | **Legacy** TP4 launcher, all four Sparks, at `max-num-seqs 64` / `max-cudagraph-capture-size 66` (66 = 11×(1+k), the first bucket covering 64). |

Every patch is guarded on `vision_n_layers > 0`: with no vision layers in the config each guarded branch resolves to the stock vLLM path, leaving a text-only run through these files untouched. Patch 4 (`spec-dspark.py`) is a **separate, required** mount and not one of the four vision files — without it the draft's always-on shared expert loads uninitialised and decode runs at roughly half speed, silently ([`../docs/PATCHES.md`](../docs/PATCHES.md), Patch 4).

## Twelve things that had to be fixed

Each was a real error, in the order it surfaced:

1. **`no module or parameter named 'aligner'`** — register the ViT, the Aligner and the four learned embeddings (`image_start/end/newline/pad`), and teach the weights mapper their prefixes.
2. **`KeyError: aligner.gate_up_proj.bias`** — the fused-MLP `stacked_params_mapping` rewrites any `.w1`/`.w3` into `gate_up_proj`; the ViT MLP and aligner legitimately use `w1`/`w2`, so guard them into the generic loader.
3. **`KeyError: layers.0.ffn.gate.e_score_correction_bias`** — the per-layer gate bias, including on the hash-MoE layers vLLM skips ("hash MoE doesn't use e_score_correction_bias" — untrue of this checkpoint, which carries one there too). Plus `bias_vl`.
4. **`'DeepseekV4Config' object has no attribute 'image_token_index'`** — the DSpark proposer expects the standard VLM field once the model reports multimodal; publish it from the tokenizer's `<｜deepseek_image｜>` id (129264).
5. **`Target model does not have 'model' attribute`** — the proposer calls `get_language_model()` then `.model`/`.lm_head`. Standard VLMs keep the tower beside a separate `language_model`; here the ViT lives *inside* the decoder stack, so `get_language_model()` returns `self`.
6. **`is not a multimodal model`** — vLLM answers `is_multimodal_model` from a **static architecture-name table**, never from the class, so `SupportsMultiModal` in the MRO is not enough. A `DeepseekV4VForConditionalGeneration` alias in `_MULTIMODAL_MODELS` points at the same class and is selected via `--hf-overrides`.
7. **Same error, still** — vLLM caches model inspection in `$VLLM_CACHE_ROOT/modelinfos/` keyed by **module + class**, and both names resolve to the same class, so the alias reused the stale pre-patch text-only verdict. Clear `modelinfos/` after changing model interfaces.
8. **`IndexError: list index out of range` in `_merge_mm_kwargs`** — ragged per-image fields need `MultiModalFieldConfig.flat_from_sizes`, not `batched`. Same shape as `deepseek_vl2.py`.
9. **Processor received 0 images** — vLLM passes `mm_data['images']` (plural, HF convention); reading `'image'` silently yielded nothing.
10. **`0 prompt placeholders`** — vLLM sets `is_update_applied=True` on the text+mm path and only *searches* the returned ids, so the placeholder must expand to the full block **inside `_call_hf_processor`** — also the only place the position-dependent `COMPRESS_PAD_TO` alignment can be expressed, since a replacement callable only receives `item_idx`.
11. **`mat1 and mat2 shapes cannot be multiplied (3x196 and 588x1024)`** — `flat_from_sizes` returns one concatenated tensor, not a per-image list; iterating it walked individual 14x14 patches into the patch embedder. Split on `num_patches`.
12. **`DeepSeek V4 hash MoE routing requires input_ids`** — the first `num_hash_layers` MoE layers route by **token id**, but the multimodal path passes `inputs_embeds` with `input_ids=None`. Fixed with `requires_raw_input_tokens = True`, which keeps the raw ids alongside the embeddings.

## Measured (2x DGX Spark GB10, TP2, temperature 0)

| | |
|---|---|
| **Vision, 112x112 image** | correct on colour *and* side, both orientations |
| **Vision, 336x336 + 26-token answer** | 1.03 s end to end |
| **Image block size** | 112x112 → 117 tokens · 168x168 → 143 · 336x336 → 129 |

Image-block token counts match the reference math. Spot-checks at temperature 0: red-left / blue-right at 112x112 → *"Red is on the left side. Blue is on the right side."*; green-top / yellow-bottom at 168x168 → *"Green and Yellow. The split is: Horizontal. The color on top is: Green."*

**Speculative depth.** Every supported path passes `k=5` through `--speculative-config`. `MTP_NUM_TOKENS=5` is set by the **TP4 launcher only** and is not the knob to copy into TP2 ([`SPARKRUN-PARITY.md`](../docs/SPARKRUN-PARITY.md) §5 item 7). Any `k=3` comparison run **without** the Patch 4 mount measures the loader rather than the drafter, because an uninitialised shared expert collapses acceptance to the loader's own signature; the k-selection evidence lives in [`PATCHES.md`](../docs/PATCHES.md) (Patch 4 Configuration). Counting prompts are draft-acceptance ceilings rather than throughput, and the KV pool is a per-boot figure — [`BENCHMARKS.md`](../docs/BENCHMARKS.md).

### The limiter here is per-step cost, not speculation

Acceptance is high on predictable work, so the headroom on this build sits in decode step cost rather than in the drafter. Two candidates: **vision tax** — this build sets `requires_raw_input_tokens = True`, so the runner slices and passes raw token ids every forward step (the text-only path does not) and the ViT plus aligner stay resident on both ranks; and **context ceiling** — a larger `max_model_len` means larger DSA indexer buffers per step. A single reload at a smaller context with everything else held constant separates the two.

## Known deviations from the reference — read before claiming parity

Two, both quality-affecting and neither crash-causing. They are why this ships as a working deployment rather than a parity claim.

**1. Bidirectional attention within image spans is not implemented.** The reference computes `get_image_visible()` and widens the sparse-attention window so tokens inside an `[IMAGE_START, IMAGE_END]` span attend bidirectionally. Here image tokens use the standard causal sparse pattern, so a token sees at most 128 of a span up to 384 tokens, and no later patches at all. The closest measured analogue (vLLM [#40106](https://github.com/vllm-project/vllm/issues/40106), Gemma-4) shows KL 0.03–0.09 concentrated at image positions. Expect degradation on dense-intra-image work — OCR, charts, documents — and note it still passes a smoke test.

**2. `bias_vl` is loaded but not applied.** Image tokens route through the text gate bias. Because every image slot carries the same placeholder id, hash routing also sends them all to one expert — which is very likely the reason `bias_vl` exists. Correct handling needs modality threaded into the MoE gate.

Neither affects text-only requests: with no image tokens the code path is the stock vLLM one.

## Prior art

vLLM issue [#54561](https://github.com/vllm-project/vllm/issues/54561) and draft PR [#54566](https://github.com/vllm-project/vllm/pull/54566) cover a parallel implementation validated on 2x RTX PRO 6000; their fixes for the hash-routing guard, OOV sentinels and `bias_vl` are worth reading, and this port reaches the same conclusions independently on several points. **What is new here is GB10 / DGX Spark**: the [NVIDIA forum position](https://forums.developer.nvidia.com/t/deepseek-v4-flash-vision-exp-is-released-as-open-weights/381911) is that the native vLLM vision processor does not work with this model, and the standing answer for vision on Spark is a sidecar VLM rather than a port.
