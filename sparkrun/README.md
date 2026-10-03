# sparkrun — the primary run path

One command deploys this repo's golden recipe. The recipes here are **1:1 matches** with the
shell launchers — same patch set, same `vllm serve` argv, same environment — and that parity is
enforced in CI. If you are not sure which to use, use sparkrun.

| Recipe | Matches | Nodes |
|---|---|---|
| [`ds4-vision-exp-tp2_v1.yaml`](ds4-vision-exp-tp2_v1.yaml) | `scripts/launch/ds4-vision-tp2.sh` | 2 (TP2) |
| [`ds4-vision-exp-tp4_v1.yaml`](ds4-vision-exp-tp4_v1.yaml) | `scripts/launch/ds4-vision-tp4.sh` | 4 (TP4) |

The full parity list — every payload mutation, serve flag, env var and container setting, plus
each deliberate deviation and why — is [`../docs/SPARKRUN-PARITY.md`](../docs/SPARKRUN-PARITY.md).
Read it before editing a recipe.

## Run it

Once per cluster:

```bash
uvx sparkrun setup     # wizard: cluster, SSH mesh, ConnectX-7 detection
```

Then:

```bash
sparkrun run ./ds4-vision-exp-tp2_v1.yaml        # or ds4-vision-exp-tp4_v1.yaml
```

No image build and no private registry: the recipe pins a public base by **digest**, fetches
this repo at a pinned commit during `pre_exec`, and rebuilds the runtime inside the container via
[`../scripts/build/stage-dspark-runtime.sh`](../scripts/build/stage-dspark-runtime.sh) — overlay,
NVFP4 stage A/B/C, Patches 3/4/6, and the vision port. Each step is verified before the server
starts, so a patch that failed to land aborts the launch instead of silently serving a half-speed
or vision-less runtime. sparkrun distributes the image and the checkpoint (pinned to the
`CURRENT.md` revision) to every node and launches the workers headless with the API on `:8888`.

Budget ~200 GB free disk per node, ~9 min for a cold first boot, ~5 min warm. `Ctrl+C` detaches
without killing the job; `sparkrun logs` / `status` / `stop` manage it afterwards.

## Verify the boot

```bash
sparkrun logs ds4-vision-exp-tp2_v1 | grep -E "stage-runtime|FATAL"
sparkrun logs ds4-vision-exp-tp2_v1 | grep "Using 'B12X' Mxfp4 MoE backend"
curl -fsS http://<head-ip>:8888/v1/models   # deepseek-v4-flash-dspark · max_model_len 1048576
```

A missing B12X line means the silent half-speed fallback. The complete post-boot checklist,
including the patch-preflight to run on every rank, is §6 of
[`../docs/SPARKRUN-PARITY.md`](../docs/SPARKRUN-PARITY.md).

Then send one text request **and one `image_url` request** — the image path is what proves the
vision port and the multimodal registry alias are live.

## Overrides you are likely to want

```bash
sparkrun run ./ds4-vision-exp-tp2_v1.yaml --dry-run                    # show the plan, change nothing
sparkrun run ./ds4-vision-exp-tp2_v1.yaml --image <fleet-image-ref>    # skip in-container staging
sparkrun run ./ds4-vision-exp-tp2_v1.yaml -o model=/mnt/models/DeepSeek-V4-Flash-Vision-Exp   # pre-placed weights
sparkrun run ./ds4-vision-exp-tp2_v1.yaml -o env.NCCL_IB_HCA=rocep1s0f0   # pin fabric if detection is wrong
```

Do **not** lower `--max-cudagraph-capture-size` below `--max-num-seqs` or off the `1+k`
multiple, and do not set `--override-generation-config`/`repetition_penalty` on this path.
Why, in both cases: [`../docs/LAUNCH-FLAGS.md`](../docs/LAUNCH-FLAGS.md).

## Benchmark

```bash
sparkrun benchmark ./ds4-vision-exp-tp2_v1.yaml --skip-run
```

Runs depth 0/32K × concurrency 1/2/6 against a running server on the coding corpus (the
llama-benchy default is a novel, which under-reports this model for code). Warm up first, and
ignore depth-32K cells at concurrency >1 unless you raise `-b tg=1024`. How to read any number
you get — warm-up decay, why streaming under-reports under spec decode, why KV pool size is a
per-boot figure — is [`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

For a single-shot throughput check:

```bash
../scripts/bench/speedtest-starfall.sh http://<head-ip>:8888
```

## Gotcha worth knowing

These recipes deliberately define **no** `post_exec`/`post_commands`. sparkrun v0.2.40 runs post
hooks before it begins following logs, gated on a hardcoded ~4-minute port-readiness timeout;
this model needs ~5–6 min to open the port, so a post hook reports `Server port never became
ready` against a cluster that booted correctly.
