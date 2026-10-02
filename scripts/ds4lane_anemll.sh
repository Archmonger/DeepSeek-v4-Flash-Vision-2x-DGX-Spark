#!/bin/bash
# ds4lane_anemll.sh — boot ONE node of Lane B on the Anemll vLLM 0.25.2 runtime.
#   usage: ds4lane_anemll.sh <0|1>     (0 = head Reddie .2, 1 = worker Spark4 .4)
#
# WHY: our production runtime is vLLM 0.21.1rc1. Anemll ships 0.25.2 (torch 2.11 / cu13)
# and still carries dspark + nvfp4_ds_mla + draft_sample_method. Newer kernels change
# STEP TIME, which is the only lever that lifts both burst and sustained decode.
#
# This image has ENTRYPOINT ["vllm","serve"] and NO baked CMD, so we pass the full arg
# list ourselves — which also means none of the baked-CMD landmines from ds4lane.sh
# (wrong VLLM_HOST_IP, per-node rank drift, missing garble fix) apply here.
#
# LANE A (Tony's work lane, Bluey+Asusi) IS NOT TOUCHED BY THIS SCRIPT.
set -euo pipefail
RANK="${1:?usage: ds4lane_anemll.sh <0|1>}"
IMG=ghcr.io/anemll/dspark-vllm-gx10:0.1.1
NAME=vllm_ds4_b_anemll
MASTER=192.168.192.2
MPORT=29625
PORT=8889
# Weights path is deliberately required: this lane serves whatever checkpoint you
# point it at, and the right path varies per node (`--model` accepts a local dir).
MODEL="${MODEL:?set MODEL to the weights directory this lane should serve}"

# --- auto-detect this node's fabric identity (non-uniform HCA names across this fleet) ---
MYIP=$(ip -o -4 addr show | grep -oE '192\.168\.192\.[0-9]+' | head -1)
IFACE=$(ip -o -4 addr show | awk -v ip="$MYIP/" '$4 ~ ip {print $2}' | head -1)
# `|| true` is REQUIRED: ibdev2netdev exits non-zero on this fleet, and with
# `set -o pipefail` that aborts the whole script right after HCA is assigned —
# the assignment succeeds, then the shell dies. Same guard as ds4lane.sh.
HCA=$(ibdev2netdev 2>/dev/null | awk -v i="$IFACE" '/\(Up\)/ && $5==i {print $1; exit}') || true
# NOTE: must be a real `if`, not `[ ... ] && ...` — under `set -e` a false test as the
# last command of a top-level && list exits the script (silently, with status 1).
if [ -z "${HCA:-}" ]; then
  HCA=$(ibdev2netdev 2>/dev/null | awk '/\(Up\)/{print $1; exit}') || true
fi
: "${HCA:?no Up RoCE device found}"
echo "node $MYIP rank=$RANK iface=$IFACE hca=$HCA"

# tunables
GMU="${DS4_GMU:-0.78}"
SEQS="${DS4_SEQS:-6}"
K="${DS4_K:-5}"
MAXLEN="${DS4_MAXLEN:-1048576}"

ARGS=(
  "$MODEL"
  --served-model-name deepseek-v4-flash-dspark
  --host 0.0.0.0 --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size 2 --pipeline-parallel-size 1
  --kv-cache-dtype nvfp4_ds_mla --block-size 256
  --max-model-len "$MAXLEN"
  --max-num-seqs "$SEQS"
  --max-num-batched-tokens 8192
  --gpu-memory-utilization "$GMU"
  --enable-prefix-caching
  --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\"}"
  --tokenizer-mode deepseek_v4
  --distributed-executor-backend mp
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice
  --reasoning-parser deepseek_v4
  --default-chat-template-kwargs '{"thinking":false}'
  --nnodes 2 --node-rank "$RANK"
  --master-addr "$MASTER" --master-port "$MPORT"
)
if [ "$RANK" != "0" ]; then
  ARGS+=(--headless)
fi

# --- lever: re-enable torch.compile + capture the real decode batch size ---
# The image defaults to CompilationMode.NONE (torch.compile OFF, breakable CUDA graphs)
# and captures [1,2,4,8,16,24,32,40,48,56,64,72] — which omits 36, and 36 is exactly
# max_num_seqs 6 x (k5 + 1), our steady-state decode batch. Both are candidate causes of
# the ~9% step-time deficit vs the B12X runtime. Set DS4_COMPILE=1 to test them together.
COMPILE_ENV=()
if [ "${DS4_COMPILE:-0}" = "1" ]; then
  CAP="${DS4_CAPTURE_SIZES:-[1,2,4,8,16,24,32,36,40,48,56,64,72]}"
  ARGS+=(--compilation-config "{\"cudagraph_capture_sizes\":$CAP}")
  COMPILE_ENV=(-e VLLM_USE_BREAKABLE_CUDAGRAPH=0)
  echo "  lever: torch.compile ON, capture sizes $CAP"
fi

docker rm -f "$NAME" 2>/dev/null || true
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g \
  --ulimit memlock=-1 --restart unless-stopped \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  --name "$NAME" \
  -e VLLM_HOST_IP="$MYIP" \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache-anemll \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/huggingface/torchinductor-anemll \
  -e TORCH_EXTENSIONS_DIR=/cache/huggingface/torch_extensions-anemll \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_SKIP_INIT_MEMORY_CHECK=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e NCCL_IB_HCA="$HCA" -e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE" \
  -e NCCL_IB_DISABLE=0 -e NCCL_NET=IB -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_IB_ADDR_RANGE=192.168.192.0/24 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=0 \
  -e NCCL_P2P_DISABLE=1 -e NCCL_NVLS_ENABLE=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  "${COMPILE_ENV[@]}" \
  "$IMG" "${ARGS[@]}"

echo "launched $NAME (rank $RANK) on $MYIP"
