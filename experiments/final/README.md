# Launch verification of scripts/serve_final.sh (2026-10-25 00:24-00:31 +07)
- `scripts/power.sh reset` -> 400 W, clocks unlocked; `scripts/restart.sh final` -> up after 282 s (fresh compile cache for the new wrapper path).
- Process arguments identical to the process that produced OFFICIAL attempt 20261024T164844-c109c6 (only the wrapper file path differs: tools/gray -> serving/). Env THINK_BUDGET=3072, HF_HUB_OFFLINE=1. vLLM selected CutlassInt8ScaledMMLinearKernel; KV cache 502,724 tokens.
- `grader.py health`: PASS stream+usage, PASS tool calling (qwen3_coder), PASS NVML energy counter; served model id `mango`.
- Local val `20261025-002935-final-verify`: 144/144 turns, 100/100 per arrival, eta 2.19x, 209.9 J/req, p95 TTFT 1.05 s, p95 TPOT 32.8 ms, accuracy 81% (agent 5/20), local S 81.6 (A 1.00). Consistent with E045 val (100/100, 2.13-2.19x).
