# Patch A — drafter-private cudagraph capture sizes (optional)

This directory holds the single patched file of this patch: `v1/spec_decode/dspark_proposer.py`,
bind-mounted read-only over `vllm/v1/spec_decode/dspark_proposer.py` on **both** nodes and
enabled with `VLLM_DSPARK_DRAFT_CAPTURE_SIZES=1`. Off by default; not mounted by the launchers.

Problem, fix, measured gain, memory cost, apply steps and status all live in
[`docs/PATCHES.md`](../../docs/PATCHES.md) — section
"Patch A (optional) — drafter-private cudagraph capture sizes".
