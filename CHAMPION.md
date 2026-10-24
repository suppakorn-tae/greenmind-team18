# Champion: E016 (since 2026-10-24 15:37 +07) — local; official run pending
Start: `configs/champion.sh` at default power (`greenmind-kit/power.sh reset` → 400 W, no clock lock).
```
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda   # ninja/nvcc for FlashInfer JIT
vllm serve CMKL/MANGO1.5-Qwen3.5-9B --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --long-prefill-token-threshold 2048 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"CMKL/MANGO1.5-Qwen3.5-9B","method":"mtp","num_speculative_tokens":2}'
```
vLLM 0.32.0, original BF16 weights only (no derived weights → nothing to push to HF).

| run | S | A | eta vs base | J/req | SLO-met | p95 TTFT | p95 TPOT | acc |
|---|---|---|---|---|---|---|---|---|
| local val #1 / #2 | 30.8 / 21.9 | 1.00 / 0.74* | 1.22x / 1.21x | 324 / 313 | 68 / 63 of 100 | 1.27 / 1.28 s | 54 / 56 ms | 80% |
| local final | 17.3 | 1.00 | 1.19x | 213 | 112/300 | 4.37 s | 87 ms | 81.0% |

Previous champion E008 (= E016 with MAX_BT 8192, no prefill cap):
**OFFICIAL val S 31.5**, A 1.00, eta 1.27x, 333 J/req, 65/100, p95 TTFT 2.02 s, p95 TPOT 38 ms, acc 88.9%;
local val eta 1.22/1.13, local final S 15.5. `configs/E008.sh` — the officially confirmed fallback.
*local val A is depressed by the agent class (n=20, scores 3–7/20 across identical runs).
