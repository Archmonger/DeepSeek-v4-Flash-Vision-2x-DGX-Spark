#!/bin/bash
RANK="${1:?usage: ds4run.sh <0|1>}"
IMG=vllm-dspark-runtime:mia-raf-pr1-nvfp4-probe-c-keys-concurrency-p2b
SCRIPT=$(docker inspect "$IMG" --format '{{index .Config.Cmd 1}}')
if [ "$RANK" = "1" ]; then
  SCRIPT="$(echo "$SCRIPT" | sed 's/--node-rank 0/--node-rank 1/') --headless"
fi
docker rm -f vllm_ds4 2>/dev/null
docker run --gpus all -d --privileged --network host --ipc host --shm-size 10g --ulimit memlock=-1 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$HOME/.cache/huggingface:/cache/huggingface" \
  --name vllm_ds4 \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e VLLM_USE_B12X_MOE=1 -e VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256 \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA=rocep1s0f0 -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_SOCKET_IFNAME=enp1s0f0np0 -e GLOO_SOCKET_IFNAME=enp1s0f0np0 -e TP_SOCKET_IFNAME=enp1s0f0np0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash \
  "$IMG" -lc "$SCRIPT"
echo "launched vllm_ds4 rank=$RANK rc=$?"
