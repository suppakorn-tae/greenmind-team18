#!/usr/bin/env bash
# GPU power policy. Rules: must be held constant for the WHOLE graded run — set it, then leave it.
#   ./power.sh 250            # power cap 250 W
#   ./power.sh 250 1215       # cap + lock SM clock at 1215 MHz
#   ./power.sh reset          # back to default limit, unlock clocks
#   ./power.sh show
set -euo pipefail
GPU=${GPU:-0}
q() { nvidia-smi -i "$GPU" --query-gpu="$1" --format=csv,noheader,nounits | head -1 | cut -d. -f1; }

sudo nvidia-smi -i "$GPU" -pm 1 >/dev/null
case "${1:-show}" in
  show) ;;
  reset) sudo nvidia-smi -i "$GPU" -pl "$(q power.default_limit)"; sudo nvidia-smi -i "$GPU" -rgc ;;
  *) sudo nvidia-smi -i "$GPU" -pl "$1"
     if [ -n "${2:-}" ]; then sudo nvidia-smi -i "$GPU" -lgc "$2,$2"; else sudo nvidia-smi -i "$GPU" -rgc >/dev/null; fi ;;
esac
nvidia-smi -i "$GPU" --query-gpu=power.limit,power.draw,clocks.sm,clocks.mem,temperature.gpu --format=csv
