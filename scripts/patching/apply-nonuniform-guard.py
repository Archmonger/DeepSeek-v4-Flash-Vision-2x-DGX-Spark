#!/usr/bin/env python3
"""Apply the non-uniform-batch speculation guard to ANY DSpark proposer version.

Why this exists: the concurrency-crash fix is a source patch to
`dspark_proposer.py`, shipped baked into this repo's runtime image from
`recipe/overlay/`. If you run a DIFFERENT prebuilt vLLM image, its `propose()`
signature may differ (e.g. some builds carry an extra `req_ids` kwarg), so
copying another version's proposer over it will crash with
`propose() got an unexpected keyword argument ...`.

This script instead patches YOUR image's own proposer in place, so the guard is
always signature-correct. Run it against the file extracted from your image:

    docker create --name t <your-image>
    docker cp t:/opt/env/lib/python3.12/site-packages/vllm/v1/spec_decode/dspark_proposer.py ./myproposer.py
    docker rm t
    python3 scripts/patching/apply-nonuniform-guard.py ./myproposer.py
    # then bind-mount ./myproposer.py at the same container path (see README).

Idempotent; verifies the result compiles.
"""
import sys, py_compile

ANCHOR = "        batch_size = self._batch_size(next_token_ids)"
MARKER = "_nonuniform_step_warned"
GUARD = ANCHOR + """

        # Non-uniform batch guard: mixed prefill/decode batches flatten to
        # unequal per-request rows, which the uniform DSpark draft kernels
        # cannot draft. Skip speculation for this step (plain decode) instead
        # of raising and killing the worker (repo concurrency>=2 fix).
        _hr = int(target_hidden_states.shape[0])
        _pr = int(target_positions.reshape(-1).shape[0])
        if batch_size > 0 and (_hr % batch_size != 0 or _pr % batch_size != 0):
            if not getattr(self, "_nonuniform_step_warned", False):
                logger.warning(
                    "DSpark proposer: non-uniform flattened batch (%d hidden "
                    "rows / %d positions for batch_size=%d); skipping "
                    "speculation for this step.", _hr, _pr, batch_size)
                self._nonuniform_step_warned = True
            return next_token_ids.new_zeros((batch_size, self.num_speculative_tokens))"""


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: apply-nonuniform-guard.py <path-to-dspark_proposer.py>", file=sys.stderr)
        return 2
    path = sys.argv[1]
    src = open(path).read()
    if MARKER in src:
        print(f"[guard] already applied to {path}; nothing to do.")
        return 0
    if ANCHOR not in src:
        near = [l for l in src.splitlines() if "batch_size" in l and "propose" not in l][:5]
        print(f"[guard] ERROR: anchor not found in {path}. vLLM version too different?\n"
              "  This proposer may not use the flatten-by-request path. Nearby:\n    "
              + "\n    ".join(near), file=sys.stderr)
        return 1
    if "logger" not in src.split("def propose")[0]:
        print(f"[guard] ERROR: no module-level `logger` in {path}; add one or edit the guard.",
              file=sys.stderr)
        return 1
    open(path, "w").write(src.replace(ANCHOR, GUARD, 1))
    py_compile.compile(path, doraise=True)
    print(f"[guard] applied to {path}; verified it compiles.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
