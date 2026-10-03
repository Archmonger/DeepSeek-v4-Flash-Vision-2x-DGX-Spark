#!/usr/bin/env bash
# One-verb DS4 update+relaunch: pull latest fixes, stop, start (both nodes).
# Preserves your local .env.dspark and any local docker-compose overrides.
# Run from the HEAD node checkout dir.
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"

echo "[update] stashing local config so pull is clean..."
STASHED=0
if ! git diff --quiet || ! git diff --cached --quiet; then
  git stash push -q -m "ds4-local-config-$(date -u +%Y%m%dT%H%M%SZ)" -- docker-compose.dspark.yml .env.dspark 2>/dev/null || true
  STASHED=1
fi

echo "[update] git pull..."
git pull --rebase --autostash

if [ "$STASHED" = "1" ]; then
  echo "[update] restoring local config..."
  git stash pop -q 2>/dev/null || {
    echo "[update] WARNING: stash pop had conflicts — resolve docker-compose.dspark.yml manually, then re-run." >&2
    exit 2
  }
fi

echo "[update] stop → start..."
bash scripts/serve/stop-deepseek-v4-flash-dspark.sh || true
sleep 3
exec bash scripts/serve/start-deepseek-v4-flash-dspark.sh
