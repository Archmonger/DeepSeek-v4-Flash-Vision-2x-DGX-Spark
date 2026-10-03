# Scripts

Every runnable script in this repo lives here, one directory per job. Payloads are **not**
scripts and stay where their consumers expect them:

| payload | lives in | consumed by |
|---|---|---|
| vLLM overlay sources + Dockerfiles | `recipe/` | `scripts/build/` |
| patch files and patchers | `patches/` | the runtime / `scripts/patching/` |
| vision-port sources and patchers | `vision-exp/port/` | `scripts/build/build-ds4v-files.sh` |
| sparkrun recipes | `sparkrun/*.yaml` | sparkrun |

Scripts resolve the repo root from their own location (`REPO_ROOT="$(cd "$(dirname
"$0")/../.." && pwd)"`), so they work from any working directory but assume the directory
depth below. Moving a script to a different depth means fixing that resolver — and for the
launchers, re-running `scripts/check/check-current.sh --write`.

## Layout

| dir | what is in it |
|---|---|
| [`launch/`](launch/) | **legacy** shell launchers — `ds4-vision-tp2.sh <0\|1>`, `ds4-vision-tp4.sh <0\|1\|2\|3>`. Kept for the unpublished local fleet image and as the argv baseline the parity check runs against; how to operate them is [`docs/LEGACY-LAUNCHERS.md`](../docs/LEGACY-LAUNCHERS.md). Serving itself is [`sparkrun/`](../sparkrun/README.md) |
| [`build/`](build/) | `stage-dspark-runtime.sh` (the in-container payload assembly the sparkrun recipes call), `build-dspark-vllm-runtime.sh` (overlay + NVFP4 stage A→B→C chain for the local fleet image), `build-ds4v-files.sh` (generates the four vision-port bind-mount files per image, for the legacy path), `verify-overlay-sources.sh` (every `COPY` in the overlay Dockerfile has a source) |
| [`serve/`](serve/) | `prepare-dspark-model-cache.sh` (download + shard-verify the checkpoint, sync to the worker), `smoke-deepseek-v4-flash-dspark.sh` (end-to-end Chat Completions smoke) |
| [`check/`](check/) | `check-current.sh` (CI: `CURRENT.md` launcher hashes), `check-patch3.sh` and `check-patch4.sh` (fail-closed preflight, run against **both** nodes), `test-prompt-token-details.py` (CPU-only argv regression) |
| [`bench/`](bench/) | measurement harnesses — see [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) for how to read their output |
| [`diagnose/`](diagnose/) | `loop_detector.py` (reasoning loop vs heavy tail, from text alone), `capture_runtime.sh` (bundle the head+worker state of a boot for a bug report) |
| [`patching/`](patching/) | `apply-nonuniform-guard.py` — apply the non-uniform-batch speculation guard to a *foreign* prebuilt image's own proposer, in place |
| [`experimental/`](experimental/) | alternate-runtime lanes (Anemll 0.25.2, baked-CMD `docker run`). Not the supported path; do not wire these into docs or CI |

## The commands you actually run

```bash
# bring the vision stack up (primary path)
uvx sparkrun setup                                    # once per cluster
sparkrun run ./sparkrun/ds4-vision-exp-tp2.yaml       # or ds4-vision-exp-tp4.yaml

# prove the runtime got the patches, on the boot you are quoting
sparkrun logs ds4-vision-exp-tp2 | grep -E "stage-runtime|FATAL"
sparkrun logs ds4-vision-exp-tp2 | grep "Using 'B12X' Mxfp4 MoE backend"

# before a PR that touches a serving flag, a recipe or a launcher
bash scripts/check/check-current.sh --write   # records the new launcher hashes in CURRENT.md
python3 scripts/check/test-prompt-token-details.py
```

The staging and preflight scripts below the `check/` row (`check-patch3.sh`, `check-patch4.sh`)
and `build/build-ds4v-files.sh` belong to the **legacy launcher** path, where the payload is
hand-staged on every node before `docker run`. Under sparkrun that assembly is done in-container by
[`build/stage-dspark-runtime.sh`](build/stage-dspark-runtime.sh). Both paths are described in
[`docs/SPARKRUN-PARITY.md`](../docs/SPARKRUN-PARITY.md) and
[`docs/LEGACY-LAUNCHERS.md`](../docs/LEGACY-LAUNCHERS.md).

## Removed: the two-node Compose lane

`docker-compose.dspark.yml` and its `start-`/`stop-`/`status-`/`logs-`/`update-and-restart`/
`validate-dspark-config` drivers were deleted. They were the generic two-node lane from the
pre-Vision text-model era and could not load the current checkpoint: it served
`DeepSeek-V4-Flash-Vision-Exp` **without** the vision-port bind mounts or the
`--hf-overrides DeepseekV4VForConditionalGeneration` alias (so the loader fails on the missing
`aligner`), **without** the Patch 6 mount (which is bind-mount-only, not baked), and with no
`--max-cudagraph-capture-size`. Nothing in the repo parses a Compose file now — the
`vllm serve` argv regression lives against the launchers instead. Recoverable from history if a
generic Compose lane is ever wanted again.

Everything about what each patch is and how it is delivered:
[`docs/PATCHES.md`](../docs/PATCHES.md).
