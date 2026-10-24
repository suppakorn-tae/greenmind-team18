# E044 result — rejected
Dynamic SD on V1 runner (PIECEWISE graphs). KV pool still 367,563 tokens (dynamic SD itself reserves the extra memory, not the V2 runner).
val 20261024-225933-e044: 83/100, eta 1.67x, 250 J/req. final 20261024-230053-e044: S 0.0, 92/300, p95 TTFT 16.4 s, 102 running / 45 waiting / KV 98%.
Lesson: dynamic speculative decoding costs ~23% of the KV pool on this model -> final-track memory collapse; TPOT fails drop (95 vs ~122) but can't pay for it.
