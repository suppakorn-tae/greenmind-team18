# E007 — E006 + --mamba-ssm-cache-dtype bfloat16 (backlog #5)
Parent E006. One variable: GDN SSM state fp32 -> bf16 (halves per-sequence state; with MTP k=3 there are k+1 state slots per seq).
Hypothesis: frees several GB -> KV pool tokens up, more concurrent seqs at peak, fewer admission-queue TTFT fails. Watch "GPU KV cache size" vs 332,606 and peak running count. Risk: accuracy drift on long/RAG/reason (state precision). Start 14:31.
