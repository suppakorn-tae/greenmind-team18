#!/usr/bin/env bash
# Usage: tools/restart.sh <id>   — stop any vLLM server, wait for GPU idle, start configs/<id>.sh, wait for /health.
ID=$1; cd "$(dirname "$0")/.."
# stop whatever serves :8000 (any launcher), plus any leftover vllm API server
for p in $(ss -ltnpH 'sport = :8000' | grep -oP 'pid=\K[0-9]+' | sort -u) $(pgrep -f "bin/vllm serve"); do [ "$p" != $$ ] && kill "$p" 2>/dev/null; done
for i in $(seq 30); do ss -ltnH 'sport = :8000' | grep -q . || break; sleep 2; done
for i in $(seq 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
mkdir -p experiments/$ID; setsid nohup configs/$ID.sh > experiments/$ID/server.log 2>&1 < /dev/null &
for i in $(seq 100); do curl -sf localhost:8000/health >/dev/null && { echo "up after $((i*3))s"; exit 0; }
  kill -0 $! 2>/dev/null || { echo "server died"; tail -5 experiments/$ID/server.log; exit 1; }; sleep 3; done
echo timeout; exit 1
