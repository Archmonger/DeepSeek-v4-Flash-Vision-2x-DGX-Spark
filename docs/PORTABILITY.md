# Running this recipe on hardware other than the author's

Notes from a clean-room bring-up on 2× DGX Spark (GB10 sm_121a, 200G CX7) that is
not the machine this recipe was developed on. The recipe itself is correct — these
are the places where a different host trips over an assumption. Numbers below were
measured on this recipe (`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`) with the Stage-C
runtime + the nvfp4 chain, DSpark k=5 probabilistic, 1M context.

## 1. `nvfp4_ds_mla` lives in the three-stage image, not the overlay

Building only `recipe/Dockerfile.dspark-runtime-overlay` gives an image whose vLLM
rejects the KV dtype:

```
vllm serve: error: argument --kv-cache-dtype: invalid choice: 'nvfp4_ds_mla'
  (choose from auto, bfloat16, float16, fp8, fp8_ds_mla, ...)
```

The dtype comes from `recipe/nvfp4/Dockerfile.stage-{a,b,c}`, chained on top of the
overlay. A one-line note near the build instructions would save the boot cycle.

## 2. `GLOO_SOCKET_IFNAME` / `TP_SOCKET_IFNAME` are baked into the base image

The base image ships these pointing at the author's NIC. On another host that
interface is down or absent and rank init dies in:

```
RuntimeError: [enforce fail at /pytorch/third_party/gloo/gloo/transport/tcp/device.cc]
```

The shipped launchers set all three of `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`
and `TP_SOCKET_IFNAME` explicitly, so they are fine. The trap is for anyone writing
their own `docker run` or service unit from a `.env.dspark`: passing only the NCCL
one still boots into the failure above. **Set all three** (defaulting the Gloo/TP
pair to the NCCL value makes a single setting cover them).

## 3. The weights mount must contain a real directory

Serving weights already on disk needs a bind mount — the launchers mount the host
weights directory at `/models` and address `$MODEL_DIR` inside it. `MODEL_DIR` must
be a real directory inside that mount: a symlink whose target is a host path outside
it resolves to nothing the container can see, and vLLM falls back to treating the
value as a repo id:

```
huggingface_hub.errors.HFValidationError: Repo id must be in the form ...
```

## 4. systemd needs `HOME`

`.env.dspark` expands `${HOME}` (for `HF_CACHE`), and the launchers use `$HOME` for
their cache dirs under `set -u`. Under a systemd unit `HOME` is unset and the launch
aborts with `HOME: unbound variable`. Adding `Environment=HOME=/root` to the unit
fixes it.

## 5. GB10 power state after a reboot (not a recipe bug, but it looks like one)

Worth flagging because the symptom mimics a bad config. After one of our nodes
crashed and rebooted, it sat at ~22 W / 2086 MHz under load while the healthy node
ran ~42 W / 2502 MHz. Because TP=2 is lockstep, the pair ran at the slow node's pace:

| | peak decode | step latency | DSpark acceptance |
|---|---|---|---|
| degraded node in the pair | 42–43 tok/s | ~140 ms | 6.02 tok/step (already optimal) |
| after `nvidia-smi -lgc 3003` on both | **83 tok/s** | ~72 ms | unchanged |

Acceptance was perfect the whole time, which is what makes this confusing — the
drafter and the config are fine; only the clock is wrong. A quick check under load:

```bash
nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader
# asymmetry between the two nodes => this
```

## 6. Thinking is off by default, and that is measurable

`--default-chat-template-kwargs '{"thinking":false}'` is a throughput choice, not a bug.
Know it before comparing a local deployment with a hosted one: turning reasoning on moves
graded quality materially (procedural suite 0.875 → 0.979), and a low output cap makes
thinking look *worse* rather than better — every length-capped failure in our run passed
once retried at 32k. Measured tables, the `reasoning` vs `reasoning_content` field-name
gotcha, and the per-request overrides are all in
[`REASONING-MODE.md`](REASONING-MODE.md).
