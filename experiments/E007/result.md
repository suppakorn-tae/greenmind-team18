# E007 result — best so far (val), promoted pending final
Run 20261024-142829-e007 val fresh: on-time 106/148 (60/100 per arrival), fails ttft 33/tpot 10/cutoff 3, p95 TTFT 1.78 s, p95 TPOT 60 ms, 303 J/req, acc 79% (agent 0.20, long 0.6, rag 1.0, reason 0.8), A=0.74 (agent flagged: 4/20 vs base 6/20 — class is noisy 3-7/20 on BF16), local val S 14.5, eta 1.12x.
Attention block 544 -> 288 tokens, KV pool 332K -> 358K tokens; peak KV usage 98% -> 56%, waiting 12 -> 0-1. Memory hypothesis confirmed: GDN fp32 state x (k+1) spec slots was the cap.
New bottleneck: per-step compute — p95 TPOT 33 -> 60 ms with all requests admitted.
Lesson: SSM state bf16 removes the admission bottleneck; no visible accuracy loss on rag/long/thai.
