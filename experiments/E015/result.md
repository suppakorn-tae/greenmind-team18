# E015 result — rejected
val 20261024-152620-e015 (util 0.95, KV pool 495K tokens): 96/136 turns, 62/100 per arrival, eta 0.89x, 344 J/req, 34.4 kJ (vs champion ~30-31 kJ), p95 TTFT 1.80 s, p95 TPOT 52 ms, fails ttft 30/tpot 7. No OOM/errors.
Per-arrival on-time/kJ 1.80 vs champion 2.02/2.08. No val benefit (val never memory-bound after E007); not promoted to final.
Lesson: extra KV is irrelevant on val; only might matter on final — untested (val gate).
Noise note: eta uses turns; agent episode length varies (136-158 turns/run) -> use per-arrival on-time/kJ as a steadier secondary metric.
