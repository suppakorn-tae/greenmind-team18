# Champion: E017 (since 2026-10-24 15:45 +07) — OFFICIAL final S 59.0
Start: `configs/champion.sh` at default power (`greenmind-kit/power.sh reset` → 400 W, no clock lock).
```
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda   # ninja/nvcc for FlashInfer JIT
vllm serve CMKL/MANGO1.5-Qwen3.5-9B --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"CMKL/MANGO1.5-Qwen3.5-9B","method":"mtp","num_speculative_tokens":2}'
```
vLLM 0.32.0, original BF16 weights only (no derived weights → nothing to push to HF).

| run | S | A | eta vs base | J/req | SLO-met | p95 TTFT | p95 TPOT | acc |
|---|---|---|---|---|---|---|---|---|
| **OFFICIAL final** (2026-10-24 08:54 UTC) | **59.0** | 1.00 | 1.78x | 247 | 215/300 | 1.91 s | 59 ms | 88.8% |
| local val #1 / #2 / confirm | 35.7 / 13.3 / 42.2 | 0.74* / 0.34* / 1.00 | 1.44x / 1.30x / 1.34x | 311 / 338 / 334 | 80 / 78 / 80 of 100 | 1.49 / 1.50 / 1.48 s | 50 / 52 / 50 ms | 80 / 76 / 79% |
| local final | 22.7 | 1.00 | 1.30x | 209 | 117/300 | 3.88 s | 80 ms | 81.0% |

E016 (cap 2048): local final S 17.3. Previous champion E008 (= E016 with MAX_BT 8192, no prefill cap):
**OFFICIAL val S 31.5**, A 1.00, eta 1.27x, 333 J/req, 65/100, p95 TTFT 2.02 s, p95 TPOT 38 ms, acc 88.9%;
local val eta 1.22/1.13, local final S 15.5. `configs/E008.sh` — the officially confirmed fallback.
*local val A is depressed by the agent class (n=20, scores 3–7/20 across identical runs).
