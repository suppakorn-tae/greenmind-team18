# E021 — champion E017 + --kv-cache-dtype fp8 (backlog #8)
Parent E017. One variable: attention KV bf16 -> fp8 (8 full-attention layers; GDN state untouched).
Why: local final peak 133 running / KV 100% / 12 waiting -> memory-capped admission at final-track peak (148 ttft fails in peak). fp8 roughly doubles attention KV tokens.
Expect: val ~neutral (not memory-bound), final fewer peak TTFT fails. Risk: Ampere has no native fp8 -> backend fallback/slower attention; accuracy on long/RAG (32K ctx). Check KV pool size and backend in log. Start 2026-10-24T16:50:31+07:00.
