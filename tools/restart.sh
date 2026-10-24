#!/usr/bin/env bash
# Usage: tools/restart.sh <id>   — stop any vLLM server, wait for GPU idle, start configs/<id>.sh, wait for /health.
ID=$1; cd "$(dirname "$0")/.."
for p in $(pgrep -f "bin/vllm serve"); do [ "$p" != $$ ] && kill "$p" 2>/dev/null; done
for i in $(seq 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
mkdir -p experiments/$ID; nohup configs/$ID.sh > experiments/$ID/server.log 2>&1 &
for i in $(seq 100); do curl -sf localhost:8000/health >/dev/null && { echo "up after $((i*3))s"; exit 0; }
  kill -0 $! 2>/dev/null || { echo "server died"; tail -5 experiments/$ID/server.log; exit 1; }; sleep 3; done
echo timeout; exit 1
