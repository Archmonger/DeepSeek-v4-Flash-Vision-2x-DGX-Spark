#!/usr/bin/env bash
# stage-dspark-runtime.sh <src-dir> <pin>
#
# Turns a stock vLLM install into this repo's DSpark + NVFP4 + Vision-Exp runtime, IN PLACE,
# inside a running container. This is the sparkrun counterpart of
# scripts/build/build-dspark-vllm-runtime.sh: that one bakes docker image layers, this one
# performs the same file mutations with cp + the stage heredocs, so a recipe needs no image
# build and no private registry.
#
# <src-dir> is a checkout of THIS repo containing recipe/ and patches/vision-port/.
# <pin>     is the commit <src-dir> came from; it keys the idempotency markers so a re-run
#           with a new pin re-applies rather than skipping.
#
# Every step is verified before this script returns success. A missing patch is FATAL here,
# which is the whole point: the two failures it prevents (Patch 3 absent -> cold-prefill
# garble; Patch 4 absent -> half-speed drafting with perfect output) are otherwise invisible
# until production. What each patch is and does: ../../docs/PATCHES.md
#
# Idempotent: safe to re-run. Markers land in $VLLM_ROOT.
set -euo pipefail

SRC="${1:?usage: stage-dspark-runtime.sh <src-dir> <pin>}"
PIN="${2:?usage: stage-dspark-runtime.sh <src-dir> <pin>}"

VLLM_ROOT="${VLLM_ROOT:-/opt/env/lib/python3.12/site-packages/vllm}"
PY="${PY:-/opt/env/bin/python}"
MODEL_DIR="$VLLM_ROOT/models/deepseek_v4/nvidia"
REGISTRY="$VLLM_ROOT/model_executor/models/registry.py"

say() { printf '[stage-runtime] %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 0. kill(1) shim. The base image ships no kill binary; vLLM's multiproc teardown calls
#    it and the engine shutdown path dies instead of stopping.
# ---------------------------------------------------------------------------
if [ ! -x /usr/bin/kill ] && [ ! -x /bin/kill ] && [ ! -x /usr/local/bin/kill ]; then
  mkdir -p /usr/local/bin
  printf '#!/bin/bash\nbuiltin kill "$@"\n' > /usr/local/bin/kill
  chmod +x /usr/local/bin/kill
  say "installed /usr/local/bin/kill shim (image has no kill binary)"
fi

# ---------------------------------------------------------------------------
# 1. DSpark overlay + NVFP4 stage A -> B -> C
# ---------------------------------------------------------------------------
if [ -e "$VLLM_ROOT/.staged-dspark-overlay-$PIN" ]; then
  say "dspark overlay + nvfp4 stages already applied for $PIN, skipping"
else
  say "applying dspark overlay from $SRC"
  test -d "$SRC/recipe/overlay/vllm" || { say "FATAL: $SRC/recipe/overlay/vllm missing"; exit 1; }
  cp -a "$SRC/recipe/overlay/vllm/." "$VLLM_ROOT/"

  for stage in a b c; do
    df="$SRC/recipe/nvfp4/Dockerfile.stage-$stage"
    test -f "$df" || { say "FATAL: $df missing"; exit 1; }
    say "applying nvfp4 stage-$stage patch"
    # The stage Dockerfiles carry their patch as one Python heredoc so the same file can be
    # docker-BUILD baked or applied to a live container this way. Exactly one heredoc per file.
    sed -n "/<<'PY'/,/^PY$/p" "$df" | sed '1d;$d' | "$PY" -
  done

  # upstream's apply-nonuniform-guard.py is deliberately NOT run here: it targets foreign
  # prebuilt images, and stacking it on this overlay short-circuits the proposer's ragged
  # path into always-rejected drafts. See docs/PATCHES.md.

  say "verifying Patch 3 (cold-prefill garble fix)"
  grep -q "is_prefill_chunk" "$VLLM_ROOT/v1/core/sched/scheduler.py" \
    || { say "FATAL: Patch 3 missing — cold prefills would garble"; exit 1; }
  say "verifying Patch 4 (draft shared-expert loader fix)"
  grep -q "shared_experts.gate_up_proj" "$VLLM_ROOT/v1/spec_decode/dspark.py" \
    || { say "FATAL: Patch 4 missing — decode would run at half speed, silently"; exit 1; }
  say "verifying Patch 6 / upstream #54 (prompt-protection cap + SWA recycle) is present"
  grep -q "VLLM_SWA_RECYCLE_SKIPPED_BLOCKS" "$VLLM_ROOT/v1/core/single_type_kv_cache_manager.py" \
    || { say "FATAL: Patch 6 (#54) missing — long conversations would lose their prefix cache"; exit 1; }

  find "$VLLM_ROOT" -name __pycache__ -type d -prune -exec rm -rf {} + || true
  "$PY" -m py_compile \
    "$VLLM_ROOT/envs.py" \
    "$VLLM_ROOT/v1/core/sched/scheduler.py" \
    "$VLLM_ROOT/v1/spec_decode/dspark.py" \
    "$VLLM_ROOT/v1/spec_decode/dspark_proposer.py"
  touch "$VLLM_ROOT/.staged-dspark-overlay-$PIN"
  say "dspark overlay + nvfp4 stages applied and verified"
fi

# ---------------------------------------------------------------------------
# 2. Vision port — native image input
#
# The stock DeepseekV4ForCausalLM has no vision tower or aligner, so it rejects the
# Vision-Exp checkpoint with "no module or parameter named 'aligner'". The port adds the
# 32-block ViT + aligner + four learned embeddings, teaches the weights mapper their
# prefixes, adds the per-layer MoE gate bias + bias_vl, and registers a multimodal
# architecture alias that --hf-overrides selects — because vLLM answers is_multimodal_model
# from a STATIC arch-name table, not from the class.
# ---------------------------------------------------------------------------
if [ -e "$VLLM_ROOT/.staged-ds4v-port-$PIN" ]; then
  say "vision port already applied for $PIN, skipping"
else
  PORT="$SRC/patches/vision-port"
  test -f "$PORT/patch_vision.py" && test -f "$PORT/patch_registry.py" \
    || { say "FATAL: patches/vision-port missing in $SRC"; exit 1; }

  say "patching DeepseekV4ForCausalLM (ViT + aligner + mapper + gate bias)"
  "$PY" "$PORT/patch_vision.py" "$MODEL_DIR/model.py"
  say "patching the model registry (multimodal alias DeepseekV4VForConditionalGeneration)"
  "$PY" "$PORT/patch_registry.py" "$REGISTRY"
  say "installing ds4v_vision.py / ds4v_mm.py"
  cp -f "$PORT/ds4v_vision.py" "$MODEL_DIR/ds4v_vision.py"
  cp -f "$PORT/ds4v_mm.py"     "$MODEL_DIR/ds4v_mm.py"

  find "$VLLM_ROOT" -name __pycache__ -type d -prune -exec rm -rf {} + || true
  "$PY" -m py_compile "$MODEL_DIR/model.py" "$MODEL_DIR/ds4v_vision.py" \
    "$MODEL_DIR/ds4v_mm.py" "$REGISTRY"

  say "verifying the port landed"
  grep -q "aligner" "$MODEL_DIR/model.py" \
    || { say "FATAL: aligner not found in patched model.py"; exit 1; }
  grep -q "DeepseekV4VForConditionalGeneration" "$REGISTRY" \
    || { say "FATAL: multimodal alias not found in registry.py"; exit 1; }
  touch "$VLLM_ROOT/.staged-ds4v-port-$PIN"
  say "vision port patched + verified"
fi

# vLLM caches model inspection under $VLLM_CACHE_ROOT/modelinfos keyed by module+class.
# Without this the alias reuses the stale pre-patch text-only entry and the API server refuses
# images with "is not a multimodal model" even though the registry is correct.
rm -rf "${VLLM_CACHE_ROOT:?VLLM_CACHE_ROOT must be set in the recipe env}/modelinfos" || true
say "cleared modelinfos inspection cache"

# Real import check: pulls the patched model module, which pulls ds4v_vision + ds4v_mm.
# Surfaces a vLLM-API mismatch here rather than six minutes into a boot.
"$PY" -c "import vllm.models.deepseek_v4
from vllm.model_executor.models.registry import _MULTIMODAL_MODELS
assert 'DeepseekV4VForConditionalGeneration' in _MULTIMODAL_MODELS, 'alias missing'
print('[stage-runtime] import check OK: multimodal alias registered')"

# ---------------------------------------------------------------------------
# 3. torch AOTAutogradCache namespace collision
#
# torch._functorch AOTAutogradCache._get_tmp_dir() returns <inductor-cache>/aotautograd —
# the same namespace vLLM's standalone_compile artifact cache writes to. One writes <key>/ as
# a DIRECTORY, the other <key> as a FILE; when both derive the same key the makedirs hits the
# file and the worker dies with FileExistsError on every cold boot that shares a JIT cache.
# Give functorch its own subdir: load and save both go through _get_tmp_dir, so both caches
# keep working.
# ---------------------------------------------------------------------------
AUTOCACHE=/opt/env/lib/python3.12/site-packages/torch/_functorch/_aot_autograd/autograd_cache.py
if grep -q "aotautograd_functorch" "$AUTOCACHE"; then
  say "torch AOTAutogradCache fix already applied"
else
  "$PY" - "$AUTOCACHE" <<'PYFIX'
import sys
path = sys.argv[1]
src = open(path).read()
old = 'return os.path.join(cache_dir(), "aotautograd")'
new = 'return os.path.join(cache_dir(), "aotautograd_functorch")'
assert src.count(old) == 1, "anchor found %d times" % src.count(old)
open(path, "w").write(src.replace(old, new, 1))
print("patched %s: functorch AOTAutogradCache -> aotautograd_functorch" % path)
PYFIX
  "$PY" -m py_compile "$AUTOCACHE"
  say "torch AOTAutogradCache fix applied"
fi

say "runtime staged for $PIN"
