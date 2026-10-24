# E042 result — abandoned (TensorRT-LLM 1.4.0rc1 slower than vLLM; cold-start JIT stalls; agent accuracy low)
TRT-LLM 1.4.0rc1 PyTorch backend serves the model (modeling_qwen3_5 + MTP max_draft_len 2, MambaHybridCacheManagerV2, mamba ssm bf16, qwen3_5 reasoning parser, qwen3_coder tools; grader health PASS; local runs need grader --force: foreign-process check only knows vllm).
No thinking budget in these runs -> compare to vLLM E017 (val 78-80/100, eta ~1.34x).
| run | per-arrival | eta | TTFT fails | TPOT fails | p95 TPOT | agent |
|---|---|---|---|---|---|---|
| cold, GUARANTEED_NO_EVICT 20261024-223259-e042 | 23/100 | 0.63x | 73 | 19 | 81 ms | 2/20 |
| warm repeat 20261024-223508-e042-2 | 54/100 | 0.86x | 38 | 8 | 57 ms | 2/20 |
| restart: cold warm-up pass 20261024-223913-e042-warmup | 24/100 | 0.60x | 72 | 18 | 81 ms | 2/20 |
| MAX_UTILIZATION after warm-up 20261024-224118-e042-maxutil | 56/100 | 0.89x | 36 | 8 | 57 ms | 2/20 |
Findings: (1) every fresh server stalls for a whole trace on runtime JIT (first request TTFT 23 s; cute-dsl kernels compile per shape, not persisted across restarts) — an official run on a fresh server would score like the cold rows; (2) even warm it trails vLLM by ~25 on-time arrivals (TTFT fails 36 vs ~10; no per-request prefill cap); (3) agent class a consistent 2/20 (vLLM 4-6/20) — tool-call path differs, accuracy risk. (4) Rules: depends on NVIDIA binary wheels (cutlass-dsl, matmul-heuristics) — would need organizer OK anyway.
Lesson: for this hybrid GDN+MTP model on 1x A100, vLLM 0.32 is clearly the best of vLLM / SGLang / TRT-LLM.
