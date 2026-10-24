# E000 — baselines (default `vllm serve`, BF16)
Command (organizer-style default): `vllm serve CMKL/MANGO1.5-Qwen3.5-9B --port 8000 --max-model-len 32768 --dtype bfloat16 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder` (process seen running at session start; launched before this log began).
| run | track | SLO-met (per arrival) | J/req | p95 TTFT | p95 TPOT | accuracy | note |
|---|---|---|---|---|---|---|---|
| OFFICIAL baseline (2026-10-24 04:50 UTC) | val | 64/100 | 467 | 2.06 s | 48 ms | 87.2% | sT 0.51, sL 0.59, eta 0.89x vs official reference |
| local `20261024-120300-vllm-val-baseline` | val | 78/100 | 463 | 1.30 s | 50.0 ms | 81% | local val reference (S defined as 0) |
| local `20261024-124012-baseline_bf16_default` | final | 106/300 | 278 | 2.87 s | 131 ms | 81.3% | local final reference |
Repeats on the same warm server (`baseline_bf16_default#2/#3`: 98/100, TPOT 24 ms) are inflated by vLLM's prefix cache replaying the identical trace -> every later experiment is measured on a fresh server (or with prefix caching off).
