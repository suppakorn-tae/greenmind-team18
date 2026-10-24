#!/usr/bin/env bash
# Usage: tools/run.sh <track> <label> <outfile> — one grader run, survives tool interrupts, waits for completion.
setsid -w ~/vllm-env/bin/python ~/grader/grader.py run --track "$1" --label "$2" --notes "$2" > "$3" 2>&1 < /dev/null
echo "== $1 $2"; grep -E "^id:|on-time|fail_reasons|per arrival|avg_power|j_per_req|p95_t|accuracy_pct|class_acc agent|  S:|  A:|eta_ratio" "$3" | tr '\n' ' '; echo
