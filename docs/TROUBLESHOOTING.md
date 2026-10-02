# Troubleshooting

Operational reference for the failure modes that recur on this two-node DSpark serving
recipe: startup blocked on cache ownership, null-content responses that need
classifying before they get reported, JIT caches shared over NFS, output corruption
that gets blamed on the weights, and what the NVFP4 KV path actually is.

**First things first:** run [`../scripts/check-patch4.sh`](../scripts/check-patch4.sh) before
anything else — a missing Patch 4 (shared-expert) mount on any node produces half speed with no
error; see [`DSPARK-SHARED-EXPERT-FIX.md`](../DSPARK-SHARED-EXPERT-FIX.md).

## Container starts but the model never loads — HF cache ownership

**Symptom.** The container starts and then dies (or hangs) before loading weights, with no
obvious error about the model path itself.

**Cause.** The container runs as **uid 1000**. If the host HF cache directory is owned by
root — common when a download was run under `sudo`, or the directory was created by a
root-run container — the container cannot write its cache/lock files.

**Fix** (credit: [@AndreasKunar](https://github.com/AndreasKunar), issue #9 — resolved
startup with no recipe edits at all):

```bash
sudo chown -R 1000:1000 /path/to/your/hf-cache
```

Check ownership before blaming the recipe:

```bash
ls -ld "${HF_CACHE:-$HOME/.cache/huggingface}"
```

## Empty `content` with real `completion_tokens` — classify before you report

A bare "null-content rate" aggregates at least five unrelated causes, and they want different
fixes. Two fields settle which one you have. Do this before opening an issue or quoting a
rate:

| `finish_reason` | `<|im_end|>` in the raw output | what it is |
| --- | --- | --- |
| `stop` | no | a **client stop string fired inside reasoning** — the CoT restated it, so generation was decapitated before `<|im_end|>`. lm-eval sends `stop[:4]` on every request. Fix: PR #21's reasoning-aware stop guard, or `until: []` client-side. |
| `length` | no | **budget exceeded**, not a hang. Reasoning is heavy-tailed even on trivial prompts (48–440 tokens measured on `"What's 1 + 1?"`). Raise `max_tokens` and re-measure; if the rate moves with the budget it was never a non-termination. |
| `length` | no, *and* the rate does not move with budget | **genuine non-termination** — a repetition loop. Detector that works: 3 consecutive 4,000-char windows below 2% novel word-8-grams. Block-level uniqueness reads *high* on plainly looping text; do not use it. Tracked in issue #18 (B). Sampling at the checkpoint's specified `temperature 1.0` measured 18/18 terminating vs 14/36 at 0.6. |
| `stop` | n/a | you sent **`reasoning_effort:"none"` with `thinking:true`** — chat-mode prompt, thinking-armed parser. |
| `stop` | yes, and the answer is missing at the client only | the client is reading **`reasoning_content`**; the response key is **`reasoning`**. |

A sixth, rarer case: the model occasionally emits a pseudo-tag scaffold
(`<STORE_AND_RETURN> 570 </STORE_AND_RETURN>`, `<STDERR> final</STDERR>630`) as its *entire*
output and never closes `<|im_end|>`, so the parser files everything as reasoning. Measured
5/60 → 0/60 with a marker-specific fallback, with ~0.8% residual on a different tag;
tag-matching is whack-a-mole, so this is documented rather than patched. Non-streaming only in
the reported measurements. Credit @robotnurse (issue #6).

Classifying costs nothing, and it is what made issue #18 tractable.

## Sharing the HF cache over NFS between the nodes — seven JIT caches, three failures

**Symptom.** Any of the following, usually in this order as each one gets fixed:

- `torch.compile` dies at startup with a `FileExistsError` / `makedirs` race
- DeepGEMM asserts `runtime != nullptr` — stale or half-written cubins read over NFS
- the head's first engine attempt dies every boot on an **ABI-mismatched FlashInfer
  `sampling.so`**, compiled by one node and silently loaded by the other. Silent because this
  stack runs `FLASHINFER_DISABLE_VERSION_CHECK=1`. With the worker entrypoint not retrying
  after a process death, this also orphans the worker on each boot.

None of those messages mention the cache. They read like broken kernels or a broken build.

**Cause.** Downloading the checkpoint once and serving it to both nodes over NFS is
the obvious move — but seven JIT/workspace caches default to (or historically sat under) the
same tree, and then **both ranks JIT into the same directories concurrently**.

**Fix.** `docker-compose.dspark.yml` mounts a **separate, node-local** volume at
`/vllm-cache` and points all seven caches at it, independent of where `HF_CACHE` lives:

| variable | value |
| --- | --- |
| `VLLM_CACHE_ROOT` | `/vllm-cache` |
| `DG_JIT_CACHE_DIR` | `/vllm-cache/deepgemm-cache` |
| `FLASHINFER_WORKSPACE_BASE` | `/vllm-cache/flashinfer` |
| `TILELANG_CACHE_DIR` | `/vllm-cache/tilelang` |
| `TORCHINDUCTOR_CACHE_DIR` | `/vllm-cache/torchinductor-cache` |
| `TRITON_CACHE_DIR` | `/vllm-cache/triton-cache` |
| `TORCH_EXTENSIONS_DIR` | `/vllm-cache/torch_extensions` |

The host path is `JIT_CACHE_DIR` (default `${HOME}/.cache/vllm-dspark`) — **keep it on local
disk on every node**. If you already ran with a shared cache, purge the poisoned directories
once: a stale `sampling.so` survives the config change.

Credit [@antoniohlc](https://github.com/antoniohlc), issue #27, who found the whole set one
crash at a time and reproduced this repo's numbers once it was fixed. The **model weights**
in `HF_CACHE` are fine on NFS; it is only the JIT tree that must be per-node.

## Before you blame the weights (gibberish / loops / CJK drift / prompt-or-XML leakage)

If the model boots and basic prompts like `hi` work, but real agent traffic randomly turns
into repeated characters, CJK drift, leaked tool/schema XML, or channel-visible junk, do not
assume the weights are bad. Three checks come first:

1. **Concurrency safety in the proposer overlay.** Confirm the Patch 2b logic is actually
   present in `recipe/overlay/vllm/v1/spec_decode/dspark_proposer.py`: ragged
   `query_start_loc` handling must **not** depend on `num_rejected_tokens_gpu`, and the
   no-rejection path must create a zero rejected-token tensor instead of falling through to
   unsafe request reshaping. Without it, concurrent DSpark requests can mix context. See
   [`PATCHES.md`](PATCHES.md) for the derivation.
2. **Image provenance.** Confirm the running image really contains the current DSpark
   overlay. A reused local tag caused misleading failures while a differently tagged image
   built from the right sources worked. When in doubt, rebuild from the intended overlay
   commit rather than trusting a tag name.
3. **Never apply a server-side `repetition_penalty` on the DSpark spec-decode path.** It is a
   documented spec-decode crash risk (illegal memory access) and it is not a garble fix. The
   launcher runs with `--generation-config vllm` and **no** `--override-generation-config`, so
   default requests do not inherit unstable model-card sampling; explicit client request
   parameters still win. For exact deterministic curl checks, send `temperature: 0` in the
   request body.

Also clear agent/orchestrator fallback lists during validation. A model that looks fixed in
direct vLLM tests can still appear poisoned if the orchestration layer silently falls back,
reboots a session, or replays a stale prompt/tool transcript into the visible message stream.
Keep harness changes separate from model runtime validation unless you are deliberately
testing that harness.

Validation gates to run after a live fix:

```text
direct vLLM prompts: clean
direct concurrent vLLM prompts: clean
agent harness prompts: clean, DeepSeek, no fallback
MTP5 accepted-token positions 0..4 active
```

## The NVFP4 path is the Stage C padded envelope

This is the **Stage C padded NVFP4** path. It keeps DeepSeek V4's known-good 584-byte
sparse-MLA cache envelope while routing the runtime through `nvfp4_ds_mla`.

It is **not** the unresolved true-layout 416-byte NVFP4 kernel work. The true-layout
experiments were useful for diagnosis but failed past roughly 411 real prompt tokens, so they
are intentionally not presented here as a reproducible recipe.
