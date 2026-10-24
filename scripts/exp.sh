#!/usr/bin/env bash
# Usage: tools/exp.sh <id> [val|both]  — restart configs/<id>.sh, run local val (+ final), print key metrics. Detached runs.
ID=$1; MODE=${2:-both}; cd "$(dirname "$0")/.."
scripts/restart.sh "$ID" || exit 1
run() { # $1 track  $2 label  $3 outfile
  setsid nohup ~/vllm-env/bin/python ~/grader/grader.py run --track "$1" --label "$2" --notes "$ID" > "$3" 2>&1 < /dev/null &
  local p=$!; while kill -0 $p 2>/dev/null; do sleep 5; done
  echo "== $1 $2"; grep -E "^id:|on-time|fail_reasons|per arrival|avg_power|j_per_req|p95_t|accuracy_pct|class_acc agent|  S:|  A:|eta_ratio" "$3" | tr '\n' ' '; echo; }
run val "$ID" experiments/$ID/val_run.out
[ "$MODE" = both ] && run final "$ID" experiments/$ID/final_run.out
grep -oE "Running: [0-9]+ reqs, Waiting: [0-9]+ reqs, GPU KV cache usage: [0-9.]+%" experiments/$ID/server.log | sort -t: -k3 -n | tail -1
