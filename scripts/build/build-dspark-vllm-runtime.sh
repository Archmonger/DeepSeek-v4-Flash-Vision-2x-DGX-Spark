#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$REPO_ROOT/.env.dspark}"

if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

DSPARK_VLLM_IMAGE="${DSPARK_VLLM_IMAGE:-vllm-dspark-runtime:dspark-nvfp4-stage-c}"
DSPARK_BASE_IMAGE="${DSPARK_BASE_IMAGE:-vllm-dspark-runtime:mia-raf-pr1}"
WORKER_BUILD="${WORKER_BUILD:-1}"

"$REPO_ROOT/scripts/build/verify-overlay-sources.sh"

build_one() {
  local host="$1"
  local checkout="$2"
  if [ "$host" = "local" ]; then
    docker build \
      -f "$REPO_ROOT/recipe/Dockerfile.dspark-runtime-overlay" \
      -t "$DSPARK_BASE_IMAGE" \
      "$REPO_ROOT/recipe/overlay"
    docker run --rm --entrypoint /opt/env/bin/python "$DSPARK_BASE_IMAGE" -c \
      "import vllm.v1.spec_decode.dspark as d; import vllm.v1.spec_decode.dspark_proposer as p; print('dspark overlay ok', d.__name__, p.__name__)"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE" \
      -f "$REPO_ROOT/recipe/nvfp4/Dockerfile.stage-a" \
      -t "$DSPARK_BASE_IMAGE-nvfp4-a" \
      "$REPO_ROOT"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE-nvfp4-a" \
      -f "$REPO_ROOT/recipe/nvfp4/Dockerfile.stage-b" \
      -t "$DSPARK_BASE_IMAGE-nvfp4-b" \
      "$REPO_ROOT"
    docker build \
      --build-arg BASE_IMAGE="$DSPARK_BASE_IMAGE-nvfp4-b" \
      -f "$REPO_ROOT/recipe/nvfp4/Dockerfile.stage-c" \
      -t "$DSPARK_VLLM_IMAGE" \
      "$REPO_ROOT"
    docker run --rm --entrypoint /opt/env/bin/python "$DSPARK_VLLM_IMAGE" -c \
      "import vllm; print('dspark nvfp4 stage-c image ok', vllm.__version__)"
  else
    ssh "$host" "mkdir -p '$checkout'"
    rsync -az --delete "$REPO_ROOT/" "$host:$checkout/"
    ssh "$host" "cd '$checkout' && DSPARK_BASE_IMAGE='$DSPARK_BASE_IMAGE' DSPARK_VLLM_IMAGE='$DSPARK_VLLM_IMAGE' WORKER_BUILD=0 ./scripts/build/build-dspark-vllm-runtime.sh"
  fi
}

build_one local "$REPO_ROOT"

if [ "$WORKER_BUILD" = "1" ]; then
  : "${WORKER_HOST:?WORKER_HOST must be set in $ENV_FILE or environment}"
  build_one "$WORKER_HOST" "${WORKER_CHECKOUT:-${WORKER_SCRIPT_DIR:-${WORKER_DIR:-$REPO_ROOT}}}"
fi
