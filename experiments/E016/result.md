# E016 result — CHAMPION (local), official pending
Config: E008 + --max-num-batched-tokens 4096 + --long-prefill-token-threshold 2048, 400 W default.
- val #1 20261024-152946-e016 (fresh): 124/158 turns, 68/100 per arrival, eta 1.22x, 324 J/req, p95 TTFT 1.27 s, p95 TPOT 54 ms, fails ttft 24/tpot 9/cutoff 3, A 0.996, local S 30.8.
- val #2 20261024-153136-e016-2: 119/158, 63/100, eta 1.21x, 313 J/req, p95 TTFT 1.28 s, fails ttft 28/tpot 9, local S 21.9 (A 0.74 agent noise).
- final 20261024-153324-e016: S 17.3 (champion E008 15.5), A 1.00, eta 1.19x (1.15), 213 J/req (224), 112/300 per arrival (119), p95 TTFT 4.37 s, p95 TPOT 87 ms, fails ttft 187/tpot 117/cutoff 5; agent 0.30 n=60, thai 0.90, rag 0.97.
vs E008: val eta mean 1.215 vs 1.18 (tie+), TTFT tail consistently better (p95_ttft_over_slo 1.45 vs 2.7, s_L ~0.73 vs 0.56), final S +12%.
Lesson: shorter worst-case steps + per-request prefill cap removes the long-prompt head-of-line TTFT staircase; neither change alone was enough (E010, E011).
