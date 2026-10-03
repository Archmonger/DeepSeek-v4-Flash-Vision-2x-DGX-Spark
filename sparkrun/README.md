# sparkrun recipe — DeepSeek-V4-Flash-Vision-Exp, 1M NVFP4 KV, 2x DGX Spark

One-command, copy-paste reproducible deployment of this repo's setup via [sparkrun](https://github.com/spark-arena/sparkrun). No image build: the recipe pulls the public base image (digest-pinned, so it can never silently change) and rebuilds the `dspark-nvfp4-stage-c` runtime inside every container at start — overlay, stage A/B/C patches, Patches 3, 4 and 6 included, pinned to commit `f45efab` of this repo. It fail-fast-verifies all three actually landed before serving, so you cannot accidentally run the half-speed stock loader. The vision port (a 32-block ViT + aligner inside vLLM) is fetched and injected by the same `pre_exec`; nothing is staged by hand on the nodes. Port write-up: [`../vision-exp/README.md`](../vision-exp/README.md); repo entry point: [`../README.md`](../README.md).

## Run it

Once, on the head node (~200 GB free disk per node):

```bash
uvx sparkrun setup     # wizard: cluster, SSH mesh, ConnectX-7 fabric detection
```

Then:

```bash
sparkrun run ./deepseek-v4-flash-vision-exp-dspark-nvfp4-1m-vllm.yaml
```

sparkrun syncs the image (~13 GB) and model (167 GB, first time only) to both nodes, injects per-host NCCL/RoCE settings, launches the worker headless and the head with the API on port 8888. First-ever boot ~9 min, warm boots ~5 min. `Ctrl+C` detaches; `sparkrun logs` / `status` / `stop` manage it afterwards. Verify the boot:

```bash
curl -fsS http://<head-ip>:8888/v1/models   # served_model_name deepseek-v4-flash-vision-exp, "max_model_len": 1048576
sparkrun logs deepseek-v4-flash-vision-exp-dspark-nvfp4-1m-vllm | grep -E "B12X|KV cache size"
# want:  Using 'B12X' Mxfp4 MoE backend     (a missing B12X line = half-speed fallback)
#        GPU KV cache size: ~2.8M tokens at 1M ctx / 0.85 gmu (varies per boot — ../docs/BENCHMARKS.md)
```

Then send a text request, and an image request (`image_url` base64) to exercise the vision path.

## What this recipe path measures

| workload | tok/s |
|---|---:|
| Protocol Starfall prompt (the video demo; video showed 50.9 live) | **71.7** |
| bulk SQL INSERTs (~100% draft acceptance) | **82–83** |
| hard code corpus (checker.ts), single-stream | 46.3 |
| hard code corpus, aggregate at c6 | 74.8 |
| hard code corpus, single-stream at 32K depth | 45.1 |

Decode is acceptance-bound and content-driven: the spread above is one server on one config. How to read these numbers, and how to re-measure them, is [`../docs/BENCHMARKS.md`](../docs/BENCHMARKS.md).

## Speed test

```bash
./speedtest-starfall.sh http://<head-ip>:8888
```

Warms the engine (first requests after boot or ~30 min idle run ~30% slow), then measures the Starfall prompt the right way: `stream: false` + `usage.completion_tokens`. Don't count SSE chunks — under spec decode vLLM emits one chunk per decode *step*, so stream-delta counting reports steps/s and under-reads by the acceptance length.

## Benchmark

The recipe ships a coding-flavored `benchmark:` block (llama-benchy's default corpus is a novel, which under-reports this model for code work):

```bash
sparkrun benchmark ./deepseek-v4-flash-vision-exp-dspark-nvfp4-1m-vllm.yaml --skip-run
```

runs depth 0/32K × concurrency 1/2/6 (~11 min) against a running server. Corpus A is TypeScript compiler internals (conservative); switch to boilerplate-heavy HTML for the upper bound with `-b book_url=https://raw.githubusercontent.com/whatwg/html/88ae68cb961651f0f92c5d2046049f53ecdfc6cf/source`. Two caveats: warm up first, and ignore the depth-32K cells at concurrency >1 unless you raise `-b tg=1024` — at the default `tg=128` those cells measure the chunked-prefill storm, not decode.

## Two things not to change

**Do NOT swap to k=3.** The earlier "k=3 wins" A/B was measured without the Patch 4 `spec-dspark.py` mount, which silently loads the draft's shared expert uninitialised ([issue #48](https://github.com/tonyd2wild/DeepSeek-v4-Flash-Vision-Exp-DSpark-1M-NVFP4-KV-2x-DGX-Spark/issues/48)). That result is retracted; with Patch 4 mounted, k=5 wins. Details in [`../docs/PATCHES.md`](../docs/PATCHES.md) (Patch 4).

**Served id, template and caches.** This recipe serves `deepseek-v4-flash-vision-exp` where the launchers serve `deepseek-v4-flash-dspark`. The model ships no Jinja chat template (`encoding/` scripts only), so `thinking:false` is the default; `--hf-overrides` selects the multimodal registry alias (without it vLLM answers "is not a multimodal model"); compile/JIT caches are forced node-local (`/cache/runtime`) because sharing the HF cache over NFS between the two ranks races `torch.compile`. Why those hold: [`../docs/PATCHES.md`](../docs/PATCHES.md) and [`../docs/LAUNCH-FLAGS.md`](../docs/LAUNCH-FLAGS.md).
