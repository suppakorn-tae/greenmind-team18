# Champion: E008 (since 2026-10-24 14:50 +07)
Start: `configs/champion.sh` (power: default 400 W, no clock lock, `greenmind-kit/power.sh reset`).
```
vllm serve CMKL/MANGO1.5-Qwen3.5-9B --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 8192 --enable-chunked-prefill --no-enable-prefix-caching \
  --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"CMKL/MANGO1.5-Qwen3.5-9B","method":"mtp","num_speculative_tokens":2}'
```
vLLM 0.32.0, original BF16 weights only (no derived weights). PATH must include the venv bin (ninja) for FlashInfer JIT.

| | S | A | eta vs base | J/req | SLO-met | p95 TTFT | p95 TPOT | acc |
|---|---|---|---|---|---|---|---|---|
| OFFICIAL val | 31.5 | 1.00 | 1.27x | 333 | 65/100 | 2.02 s | 38 ms | 88.9% |
| local val | 21.2 | 0.74* | 1.22x | 311 | 63/100 | 1.69 s | 59 ms | 79% |
| local final | 15.5 | 1.00 | 1.15x | 224 | 119/300 | 3.66 s | 88 ms | 80.7% |
*local val A depressed by agent class noise (n=20).
