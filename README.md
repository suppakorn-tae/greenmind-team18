# GreenMind — MANGO1.5-Qwen3.5-9B on 1× A100-SXM4 40GB

Goal: maximise S = 100 · A · s_E^0.6 · s_T^0.2 · s_L^0.2 (on-time requests per GPU joule, accuracy held).

## Result
| config | track | S | A | η vs base | J/request | SLO-met | p95 TTFT | p95 TPOT | accuracy |
|---|---|---|---|---|---|---|---|---|---|
| baseline (`vllm serve`, BF16 defaults) | official val | — | 1.00 | 0.89× | 467 | 64/100 | 2.06 s | 48 ms | 87.2% |
| E008 (BF16 + MTP k=2 + bf16 SSM state) | official val | 31.5 | 1.00 | 1.27× | 333 | 65/100 | 2.02 s | 38 ms | 88.9% |
| **E017 champion** (E008 + prefill scheduling) | **official final** | **59.0** | 1.00 | 1.78× | 247 | 215/300 | 1.91 s | 59 ms | 88.8% |

Champion launch command: [`CHAMPION.md`](CHAMPION.md) / `configs/champion.sh` (400 W default, no clock lock).
Original CMKL BF16 weights only, no derived weights. Every experiment, failures included: [`EXPERIMENTS.md`](EXPERIMENTS.md),
`experiments/<id>/{card,result}.md`, machine log `experiments/log.jsonl`. Local grader runs: `~/grader/runs*`.

## Approach — three things moved the score
1. **Speculative decoding with the built-in MTP head, k=2** (per-position acceptance 0.87 / 0.74). It cuts J/request by about 30% and TPOT by about 25%.
   k=3 is a tie, and k=1 is worse because fewer tokens per step costs more than the cheaper step saves.
2. **`--mamba-ssm-cache-dtype bfloat16`.** With MTP, each sequence keeps k+1 Gated-DeltaNet recurrent states in fp32.
   That capped concurrency at about 46 sequences (KV 98%, peak queueing). The bf16 state halves it: peak KV drops to 56%,
   TTFT fails go from 59 to 33, and no accuracy loss is visible on RAG/long/Thai.
3. **Prefill scheduling: `--max-num-batched-tokens 4096 --long-prefill-token-threshold 1024`.** The remaining TTFT misses were
   short prompts stuck behind one 18–20K-token prompt's prefill, a "staircase" of TTFTs from 2.6 s down to 0.5 s. Capping each request's prefill
   chunk lets new arrivals start in the next step. Local val TTFT fails go from about 30 to about 10, and per-arrival SLO from 63 to 80 out of 100.
   Neither knob works alone, and the ratio matters: caps of 2048 and 512, and a 2048 budget, were all worse.

## What did not work (and why)
- **The provided AWQ-W4A16 checkpoint.** The agentic class fell to a deterministic 2/20 (BF16 scores 3–7/20), and Marlin W4A16 decode is slower at large batch.
- **Power caps of 250 and 300 W.** J/request drops 13–18%, but prefill is compute-bound, so TTFT fails rise and η doesn't improve.
- **Prefix caching with mamba align mode.** Only a 3% hit rate on this workload. Warm repeat runs look far better only because the trace replays into the cache.
- **Warmup script, `--gpu-memory-utilization 0.95`, and smaller chunk budget alone.** Each was neutral or worse.

## Trade-offs / caveats
- Local final is harsher than official (117/300 vs 215/300), but the rankings agree. Local val tracks official val (η 1.22 vs 1.27).
- The agent class (n=20 on val) is very noisy, so local val A swings from 0.34 to 1.0 on identical configs. Use final-track accuracy (n=60) and official A.
- vLLM 0.32.0 needs `ninja` and `nvcc` on PATH for FlashInfer's sampler JIT (set in every config).
