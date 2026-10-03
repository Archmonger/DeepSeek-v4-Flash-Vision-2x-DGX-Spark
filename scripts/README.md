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
| [`launch/`](launch/) | the supported serving entry points: `ds4-vision-tp2.sh <0\|1>`, `ds4-vision-tp4.sh <0\|1\|2\|3>` |
| [`build/`](build/) | `build-dspark-vllm-runtime.sh` (overlay + NVFP4 stage A→B→C chain), `build-ds4v-files.sh` (generates the four vision-port bind-mount files per image), `verify-overlay-sources.sh` (every `COPY` in the overlay Dockerfile has a source) |
| [`serve/`](serve/) | the Compose lane: `start-`, `stop-`, `status-`, `logs-`, `smoke-deepseek-v4-flash-dspark.sh`, `update-and-restart.sh`, `validate-dspark-config.sh`, `prepare-dspark-model-cache.sh` |
| [`check/`](check/) | `check-current.sh` (CI: `CURRENT.md` launcher hashes), `check-patch3.sh` and `check-patch4.sh` (fail-closed preflight, run against **both** nodes), `test-prompt-token-details.py` (CPU-only argv regression) |
| [`bench/`](bench/) | measurement harnesses — see [`docs/BENCHMARKS.md`](../docs/BENCHMARKS.md) for how to read their output |
| [`diagnose/`](diagnose/) | `loop_detector.py` (reasoning loop vs heavy tail, from text alone), `capture_runtime.sh` (bundle the head+worker state of a boot for a bug report) |
| [`patching/`](patching/) | `apply-nonuniform-guard.py` — apply the non-uniform-batch speculation guard to a *foreign* prebuilt image's own proposer, in place |
| [`experimental/`](experimental/) | alternate-runtime lanes (Anemll 0.25.2, baked-CMD `docker run`). Not the supported path; do not wire these into docs or CI |

## The commands you actually run

```bash
# bring the vision stack up (worker first, then head)
./scripts/launch/ds4-vision-tp2.sh 1
./scripts/launch/ds4-vision-tp2.sh 0

# prove the patches are mounted, on BOTH nodes, before quoting a number
./scripts/check/check-patch4.sh <head-container> <worker-container>

# before a PR that touches a launcher or a serving flag
bash scripts/check/check-current.sh --write   # records the new hashes in CURRENT.md
python3 scripts/check/test-prompt-token-details.py
```

Everything about what each patch is and how it is delivered:
[`docs/PATCHES.md`](../docs/PATCHES.md).
