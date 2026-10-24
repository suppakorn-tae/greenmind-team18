# E012 result — rejected alone (energy win, TTFT loss)
val 20261024-151129-e012 @250 W: on-time 91/148, 45/100 per arrival, eta 1.14x, 254 J/req (-18% vs champion 311), avg 246 W, p95 TTFT 2.90 s, p95 TPOT 56 ms, fails ttft 48/tpot 8/cutoff 3, A 1.0 (agent 7/20), S 19.5.
Power cap cuts energy as expected and doesn't hurt TPOT, but slows compute-bound prefill -> TTFT staircase gets worse (30 -> 48 fails).
Lesson: cap helps J/req; pair it with the TTFT fix (E011 chunk cap) -> E013.
