# Legacy: the shell launchers

**The primary serving path is sparkrun** — [`../sparkrun/README.md`](../sparkrun/README.md). This
page documents the predecessor: the shell launchers in
[`../scripts/launch/`](../scripts/launch/), which predate sparkrun and are still in the tree and
still work.

Three reasons this path is kept rather than deleted:

1. **The fleet image is local and unpublished.** Both launchers run
   `vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b`, which exists only on
   these machines. sparkrun pins a public base image by digest and rebuilds the payload inside the
   container; the launchers consume the prebuilt image as-is.
2. **The launchers are the parity reference.** `scripts/check/test-prompt-token-details.py` treats
   the launcher's `vllm serve` argv as the baseline and asserts each recipe reaches it — no missing
   flags, no extra flags, equal values. Remove the launcher and that check loses its reference.
3. **CI hashes them.** `scripts/check/check-current.sh` requires a `CURRENT.md` line for every
   `scripts/launch/*.sh`, so a launcher edit without a `CURRENT.md` edit fails the build.

If you just want the serving stack up, use sparkrun. Come here when you need the local image, an
air-gapped host with weights already staged, or to reproduce a report made against the launcher.

## What you have to stage yourself

The launcher copies nothing. Every node needs the payload under `/var/tmp` before the container
starts — a missing file is a hard `exit 4`, and `TP2` checks fewer files than `TP4` (see
[Preflight](#preflight-and-exit-codes)).

```bash
# run on EVERY node
./scripts/build/build-ds4v-files.sh          # -> /var/tmp/ds4v_{model,vision,mm,registry}.py
cp recipe/overlay/vllm/v1/core/sched/scheduler.py               /var/tmp/patch3-scheduler.py
cp recipe/overlay/vllm/v1/spec_decode/dspark.py                 /var/tmp/spec-dspark.py
cp recipe/overlay/vllm/v1/core/single_type_kv_cache_manager.py  /var/tmp/patch6-single_type_kv_cache_manager.py
```

Weights must already be on disk at `$MODELS_HOST/$MODEL_DIR`, or the launcher exits `3`. Which
file carries which patch, and how to verify each one landed: [`PATCHES.md`](PATCHES.md).

Under sparkrun none of this is manual —
[`../scripts/build/stage-dspark-runtime.sh`](../scripts/build/stage-dspark-runtime.sh) assembles
the same payload inside the container at boot, verifies it, and aborts rather than serving a
half-patched runtime.

## TP2 — asusi + bluey

| rank | node | fabric IP | role | weights |
|---|---|---|---|---|
| 1 | **Bluey** | `192.168.192.1` | worker (`--headless`) | local at `/var/tmp/models`, NFS-exported |
| 0 | **Asusi** | `192.168.192.3` | head — serves `:8888` | NFS from Bluey at `/mnt/bluey-models` |

**Worker first, then head.** `--master-port 25440`.

```bash
./scripts/launch/ds4-vision-tp2.sh 1     # on bluey
./scripts/launch/ds4-vision-tp2.sh 0     # on asusi
```

Run each command **on its own node** — the launcher resolves nothing remotely; it reads local
`/var/tmp` and local mounts and fails if they are absent.

Plane A only. Plane B (`roceP2p1s0f0`) is not on a common subnet between these two nodes (Asusi
is link-local `169.254.61.52`, Bluey is `192.168.193.1`), so `NCCL_IB_MERGE_NICS` would try to
bring RC queue pairs up across mismatched subnets.

## TP4 — all four Sparks

| rank | node | fabric IP | role | weights |
|---|---|---|---|---|
| 3 | **Spark4** | `192.168.192.4` | worker (`--headless`) | NFS |
| 2 | **Reddie** | `192.168.192.2` | worker (`--headless`) | NFS |
| 1 | **Bluey** | `192.168.192.1` | worker (`--headless`) | local |
| 0 | **Asusi** | `192.168.192.3` | head — serves `:8888` | NFS |

**Launch order 3 → 2 → 1 → 0**, each command on its own node. `--master-port 25460`.

```bash
./scripts/launch/ds4-vision-tp4.sh 3     # spark4
./scripts/launch/ds4-vision-tp4.sh 2     # reddie
./scripts/launch/ds4-vision-tp4.sh 1     # bluey
./scripts/launch/ds4-vision-tp4.sh 0     # asusi (head)
```

TP4 is TP2 with `--tensor-parallel-size 4`, `--nnodes 4`, `--max-num-seqs 64`,
`--max-cudagraph-capture-size 66` (`= 11 × (1+k)`, the first capture bucket that covers 64
requests), the 4-node rank map, and master-port `25460`. Flag semantics:
[`LAUNCH-FLAGS.md`](LAUNCH-FLAGS.md).

## Ambient knobs

The launchers do **not** source `.env.dspark`. Everything else comes from their own `-e` blocks;
these four are read from the ambient shell:

| variable | default | effect |
|---|---|---|
| `MODEL_DIR` | `DeepSeek-V4-Flash-Vision-Exp` | which checkpoint dir under `$MODELS_HOST`. The uncensored drop-in is `MODEL_DIR=keys-DeepSeekV4Flash-Vision-EXP-ablit` — same shards, same tokenizer |
| `PROTECTED_FRACTION` | `0.30` | Patch 6 — `VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION` |
| `SWA_RECYCLE` | `1` | Patch 6 — `VLLM_SWA_RECYCLE_SKIPPED_BLOCKS` |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` (**TP2 only**) | inference RPC deadline; the engine-ready timeout above it covers boot only (issue #8) |

## Preflight and exit codes

| exit | meaning |
|---|---|
| `1` | container exited immediately after `docker run` — read `docker logs` |
| `2` | usage error — rank outside `0..1` (TP2) / `0..3` (TP4) |
| `3` | `MODEL MISSING` — `$MODELS_HOST/$MODEL_DIR` is not a directory |
| `4` | a required staged file is missing under `/var/tmp` |

TP2 preflights three files (`patch3-scheduler.py`, `spec-dspark.py`,
`patch6-single_type_kv_cache_manager.py`) and names the source path under `recipe/overlay/` in the
error. It does **not** preflight the four vision files, so a missing one fails at container start
instead of at the gate. TP4 checks all seven in a loop (`MISSING /var/tmp/<f>` → exit `4`).

This asymmetry is a launcher wart, not a difference in what the two tiers need — both require all
seven files.

## Verify a launcher boot

```bash
./scripts/check/check-patch3.sh <head-container> <worker-container>   # cold-prefill garble guard
./scripts/check/check-patch4.sh <head-container> <worker-container>   # shared-expert mapping
curl -fsS http://<head-ip>:8888/v1/models                            # deepseek-v4-flash-dspark
```

Run the patch checks on **both** nodes. A missing Patch 4 mount costs roughly half your decode
speed while producing perfect output and raising no error — see
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md). Then send one text request and one `image_url`
request; the image path is what proves the vision port is live.

## How this path differs from sparkrun

Both paths reach the same patch set and the same `vllm serve` argv. What differs is the transport
and the coupling:

| | shell launcher | sparkrun recipe |
|---|---|---|
| image | local, unpublished, run as-is | public base pinned by digest, payload staged in-container |
| payload staging | manual, per node, before launch | automatic at boot, verified, fails closed |
| weights | bind-mounted pre-staged, fully offline | pinned HF repo + revision, distributed by sparkrun (overridable) |
| topology | hardcoded to this fleet's IPs | detected per host, overridable |
| caches | dedicated `/vllm-cache` bind | `/cache/runtime` |
| IPC namespace | `--ipc host` | `ipc: shareable` (host IPC is unsafe under sparkrun's user mapping) |

Every one of these is enumerated with its reasoning in
[`SPARKRUN-PARITY.md`](SPARKRUN-PARITY.md) §5. Read that before changing either side.

## Two launcher-only quirks to know before copying commands between tiers

The launchers are not symmetric with each other, and neither difference is obviously intentional:

- `MTP_NUM_TOKENS=5` is set on **TP4 only**.
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` is set on **TP2 only**.

Both tiers run the same `k=5` probabilistic DSpark spec, so `k` comes from `--speculative-config`
regardless. Do not read `MTP_NUM_TOKENS` as the way to set depth on TP2. Recorded as
[`SPARKRUN-PARITY.md`](SPARKRUN-PARITY.md) §5 item 7.

## Changing a launcher

1. Update [`../CURRENT.md`](../CURRENT.md) in the same PR and say which line your change moves.
2. `bash scripts/check/check-current.sh --write` — CI runs the same script without `--write`.
3. `python3 scripts/check/test-prompt-token-details.py` — if you moved a serving flag, the matching
   sparkrun recipe has to move with it or this fails.
