# E043 result — rejected
Dynamic SD [[1,64,2],[65,256,1]] + VLLM_USE_V2_MODEL_RUNNER=1 (full + piecewise graphs; acceptance length drops to 1.88 at high batch as scheduled).
val 20261024-225144-e043: 81/100, eta 1.64x (E030 1.72-1.77), 251 J/req, p95 TPOT 59 ms. final 20261024-225302-e043: S 10.7, 90/300, p95 TTFT 14.0 s, 101-107 running / 37-39 waiting / KV 95-99%.
Cause: V2 runner reserves more memory -> KV pool 367,563 tokens vs 475,367 (V1) -> final peak memory-capped. TPOT fails did drop (96 vs ~122).
