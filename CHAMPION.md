# Champion: E046 = E045 served as "mango" — OFFICIAL final S 99.38 (2026-10-24 16:48 UTC)
OFFICIAL: S 99.38, A 1.00, sE 1.00, sT 0.969, sL 1.00, eta 3.82x, 157 J/req, 294/300 on time, 0 truncated/errors, p95 TTFT 1.21 s, p95 TPOT 37 ms, accuracy 88.3%.
Caveat: longctx retention 0.71 (n=15, below the 20-question gate size). `configs/champion.sh` == `configs/E046.sh` (= E045 + `--served-model-name mango`).
**Launch the final server with `scripts/serve_final.sh`** (same flags as `configs/E046.sh`, portable paths). The command block below is the E045 record (served name `suppakorn-k/...`; E046 uses `mango`).

= E030 (OFFICIAL final 88.0) with the team's W8A8 INT8 checkpoint `suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8` (private HF repo,
rev 1909e853, made 2026-10-24 by a teammate with llm-compressor GPTQ W8A8 from the official CMKL weights; open calibration data:
WangchanThaiInstruct, UltraChat, Open-Platypus). Fallback: `configs/E030.sh` (official BF16 weights).

Start: `configs/champion.sh` at default power (400 W, no clock lock). The repo is private, so the weights must already be in the
HF cache (download once with a read token), and the config serves the local snapshot with `HF_HUB_OFFLINE=1` (no token at runtime).
```
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda HF_HUB_OFFLINE=1
M=$HOME/.cache/huggingface/hub/models--suppakorn-k--MANGO1.5-Qwen3.5-9B-W8A8/snapshots/1909e853b46b2618457ba4205e48a61df6a0d888
THINK_BUDGET=3072 ~/vllm-env/bin/python ~/repo/serving/vllm_think_budget.py serve $M --served-model-name suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8 \
  --host 0.0.0.0 --port 8000 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 --max-num-seqs 256 \
  --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 --enable-chunked-prefill \
  --no-enable-prefix-caching --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"'$M'","method":"mtp","num_speculative_tokens":2}'
```
| E045 local | val #1 / #2 | final #1 / #2 |
|---|---|---|
| SLO-met (per arrival) | 100 / 100 of 100 | 245 / 261 of 300 |
| eta vs base | 2.13 / 2.19x | 3.07 / 2.99x |
| J/req | 205 / 213 | 175 / 180 |
| p95 TTFT / TPOT | 1.11 s / 36 ms, 1.09 s / 36 ms | 1.62 s / 51 ms, 1.61 s / 48 ms |
| accuracy | 81 / 78% (agent noise) | 83.0 / 84.0%, A 1.00 |
| local S | 4.0 / 60.3 (agent-class A noise) | **93.7 / 95.0** (E030: 28.8 / 25.5) |

## E030 — OFFICIAL final S 88.0 (previous champion)
E033 (E030 + gpu util 0.96) scored OFFICIAL 81.6 despite local +12% -> reverted. `configs/champion.sh` == `configs/E030.sh`.

## E030 — OFFICIAL final S 88.0
OFFICIAL final: S 88.0, A 1.00, R 1.034, sE 0.96, sT 0.71, sL 0.85, eta 2.87x, 173 J/req, p95 TTFT 1.85 s, p95 TPOT 57 ms, 242/300, acc 90.7%.
= E027 with `THINK_BUDGET=3072` (the only change; set inside `configs/champion.sh`).
E030 local: val eta 1.77 / 1.72x, 83 / 81 of 100, 258 / 261 J/req, 0 cutoffs; final S 28.8 / 25.5, eta 1.46 / 1.39x, 123 / 114 of 300, 195 J/req, acc 81.0 / 80.7%.

## E027 (budget 5120) — previous local champion
= E017 (OFFICIAL final S 59.0) + server-side default thinking budget 5120 tokens (authorized by the team lead 2026-10-24 ~19:00;
organizer confirmation still recommended). Fallback without the budget: `configs/E017.sh`.

Start: `configs/champion.sh` at default power (`greenmind-kit/power.sh reset` → 400 W, no clock lock).
```
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda   # ninja/nvcc for FlashInfer JIT
THINK_BUDGET=3072 python ~/repo/serving/vllm_think_budget.py  # E030; E027 used 5120 serve CMKL/MANGO1.5-Qwen3.5-9B --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 \
  --enable-chunked-prefill --no-enable-prefix-caching \
  --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"CMKL/MANGO1.5-Qwen3.5-9B","method":"mtp","num_speculative_tokens":2}'
```
`serving/vllm_think_budget.py` = `vllm serve` plus one patch: chat requests that don't set `thinking_token_budget`
get 5120 (vLLM's built-in budget logic forces the reasoning end marker). The client's max_tokens, temperature and thinking mode are untouched.

| E027 (local) | val | final |
|---|---|---|
| runs | 4 | 3 |
| S | 47.2 / 39.5 / 36.7 / 51.1 | 25.2 / 25.0 / 26.6 (mean 25.6; E017 20.5) |
| eta vs base | 1.76 / 1.58 / 1.45 / 1.51x (mean 1.58; E017 1.34) | 1.38 / 1.36 / 1.41x |
| SLO-met | 84 / 80 / 82 / 83 of 100 | 118 / 115 / 120 of 300 |
| cutoffs | 0 / 0 / 1 / 0 | 0 / 0 / 0 |
| J/req | 261 / 280 / 312 / 297 | 200 / 201 / 198 |

# Previous champion: E017 (OFFICIAL final S 59.0)
| run | S | A | eta vs base | J/req | SLO-met | p95 TTFT | p95 TPOT | acc |
|---|---|---|---|---|---|---|---|---|
| **OFFICIAL final** (2026-10-24 08:54 UTC) | **59.0** | 1.00 | 1.78x | 247 | 215/300 | 1.91 s | 59 ms | 88.8% |
| local val #1 / #2 / confirm | 35.7 / 13.3 / 42.2 | 0.74* / 0.34* / 1.00 | 1.44x / 1.30x / 1.34x | 311 / 338 / 334 | 80 / 78 / 80 of 100 | 1.49 / 1.50 / 1.48 s | 50 / 52 / 50 ms | 80 / 76 / 79% |
| local final | 22.7 | 1.00 | 1.30x | 209 | 117/300 | 3.88 s | 80 ms | 81.0% |

E016 (cap 2048): local final S 17.3. Previous champion E008 (= E016 with MAX_BT 8192, no prefill cap):
**OFFICIAL val S 31.5**, A 1.00, eta 1.27x, 333 J/req, 65/100, p95 TTFT 2.02 s, p95 TPOT 38 ms, acc 88.9%;
local val eta 1.22/1.13, local final S 15.5. `configs/E008.sh` — the officially confirmed fallback.
*local val A is depressed by the agent class (n=20, scores 3–7/20 across identical runs).
