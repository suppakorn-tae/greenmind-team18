# Experiments

| id | parent | change | status | val S | final S | official S | J/req | SLO | acc | lesson |
|---|---|---|---|---|---|---|---|---|---|---|
| E000 |  | default vllm serve BF16 baseline | baseline | 0.0 | 0.0 | 0.0 | 467 | 64/100 | 87.2 | official val baseline: s_T 0.51 s_L 0.59 p95 TTFT 2.06s |
| E001 | E000 | AWQ+MTP k=3 tuned flags | rejected-cold | 0.0 |  |  | 272 | 45/100 cold; 132/138 warm | 77.0 | cold: 51 TTFT fails from Triton JIT in inference. warm: eta 1.85x, TPOT 20ms, 227 J/req. agent acc 0.1 vs baseline 0.2-0.3 (n=20, A=0.05) - suspect AWQ |
| E002 | E001 | warmup.py | rejected | 0.0 |  |  | 269 | 81/137 | 78.0 | warm repeats are prefix-cache artifacts; peak TTFT stalls from 8192-token prefill chunks |
| E003 | E002 | max-num-batched-tokens 2048 | rejected | 0.0 |  |  | 283 | 72/135 | 76.0 | smaller chunks worse; per-step fixed cost dominates, suspect MTP; ablate |
| E004 | E001 | remove MTP (ablation) | rejected | 0.0 |  |  | 315 | 80/134 | 77.0 | agent 2/20 without MTP too -> AWQ (or flags) culprit; MTP improves J/req |
| E005 | E001 | AWQ -> BF16 weights (keep MTP k=3 + flags) | running |  |  |  |  |  |  |  |

## Key findings

_(none yet)_
