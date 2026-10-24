# Experiments

| id | parent | change | status | val S | final S | official S | J/req | SLO | acc | lesson |
|---|---|---|---|---|---|---|---|---|---|---|
| E000 |  | default vllm serve BF16 baseline | baseline | 0.0 | 0.0 | 0.0 | 467 | 64/100 | 87.2 | official val baseline: s_T 0.51 s_L 0.59 p95 TTFT 2.06s |
| E001 | E000 | AWQ+MTP k=3 tuned flags | rejected-cold | 0.0 |  |  | 272 | 45/100 cold; 132/138 warm | 77.0 | cold: 51 TTFT fails from Triton JIT in inference. warm: eta 1.85x, TPOT 20ms, 227 J/req. agent acc 0.1 vs baseline 0.2-0.3 (n=20, A=0.05) - suspect AWQ |
| E002 | E001 | warmup.py | rejected | 0.0 |  |  | 269 | 81/137 | 78.0 | warm repeats are prefix-cache artifacts; peak TTFT stalls from 8192-token prefill chunks |
| E003 | E002 | max-num-batched-tokens 2048 | rejected | 0.0 |  |  | 283 | 72/135 | 76.0 | smaller chunks worse; per-step fixed cost dominates, suspect MTP; ablate |
| E004 | E001 | remove MTP (ablation) | rejected | 0.0 |  |  | 315 | 80/134 | 77.0 | agent 2/20 without MTP too -> AWQ (or flags) culprit; MTP improves J/req |
| E005 | E001 | AWQ -> BF16 (MTP k=3 + flags) | promoted-mainline | 0.0 |  |  | 298 | 87/148 | 82.0 | BF16 restores agent 7/20 (A=1.0); TTFT fails from KV-full admission queueing at peak (96.6% KV, prefix hit 3%) |
| E006 | E005 | prefix caching off | neutral | 0.0 |  |  | 328 | 86/148 | 76.0 | prefix cache irrelevant (3% hits); KV still full at 46 seqs; agent class noisy 3-7/20 on BF16 |
| E007 | E006 | mamba ssm cache bf16 | promoted | 14.5 |  |  | 303 | 106/148 (60/100) | 79.0 | bf16 SSM state frees memory: peak KV 98->56%, TTFT fails 59->33; now TPOT-bound (60ms) |
| E008 | E007 | MTP k 3->2 | champion | 21.2 | 15.5 | 31.5 | 311 | 63/100 | 79.0 | k=2 >= k=3; official S 31.5; final track TPOT-bound at high concurrency |
| E009 | E008 | MTP k 2->1 | rejected | 0.0 | 4.2 |  | 381 | 64/100 | 80.0 | k=1 worse J/req and final TPOT; final TPOT tail = 8K prefill chunks starving decode |
| E010 | E008 | max-num-batched-tokens 4096 | rejected | 18.2 |  |  | 301 | 64/100 | 80.0 | within noise; TTFT fails are head-of-line blocking of short prompts behind long prefills |
| E011 | E008 | long-prefill-token-threshold 2048 | rejected | 14.1 |  |  | 318 | 66/100, 64/100 | 79.0 | s_L 0.56->0.74 but eta mean 1.05 vs 1.22; s_E dominates |
| E012 | E008 | power cap 250 W | rejected | 19.5 |  |  | 254 | 45/100 | 82.0 | 250W: J/req -18% but TTFT fails 30->48 (prefill slower) |
| E013 | E012 | 250W + chunk cap 2048 | rejected | 14.3 |  |  | 267 | 45/100 |  | 250W too slow at peak regardless of chunking |
| E014 | E008 | power cap 400->300 W | running |  |  |  |  |  |  |  |

## Key findings

- AWQ-W4A16 checkpoint: agent class deterministic 2/20 (BF16: 3-7/20) and slower TPOT under batch (Marlin) -> rejected; BF16 is the line.
- MTP on GDN hybrid multiplies per-sequence recurrent state (k+1 slots); with fp32 state this capped concurrency at ~46 seqs (KV 98%, queueing TTFT). `--mamba-ssm-cache-dtype bfloat16` fixed it (peak KV 56%, TTFT fails 59->33), no visible accuracy loss.
- MTP k=2 >= k=3 (acceptance 0.87/0.74/0.59 per position).
- Prefix caching: 3% hit rate on this workload; align mode no benefit -> off.
- Smaller prefill chunks (2048) hurt: per-step fixed cost.
- Repeat runs on one server with prefix caching are inflated (warm cache replays) — always fresh server, or prefix caching off.
- Local val eta tracks official (1.22 vs 1.27); local val A is pessimistic because the agent class (n=20) is noisy.

