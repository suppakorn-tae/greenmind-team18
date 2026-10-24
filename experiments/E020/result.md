# E020 result — rejected (val tie, final collapse)
- val 20261024-161238-e020: 123/148, 75/100, eta 1.33x, 296 J/req, p95 TTFT 1.95 s, p95 TPOT 52 ms, fails ttft 15/tpot 8.
- val #2 20261024-161417-e020-2: 123/148, 75/100, eta 1.37x, 287 J/req, fails ttft 17/tpot 8.
  -> eta tie with E017 (1.35 vs 1.36 mean), J/req -10% consistent, SLO -4 consistent -> final run (gate: same S, lower J).
- final 20261024-161556-e020: S 2.9, A 1.00, eta 1.01x, 98/300, p95 TTFT 19.8 s (!), p95 TPOT 51 ms, fails ttft 241/tpot 23, 208 J/req, acc 82.3%.
Diagnosis: final peak 82 running / 51 waiting / KV 98.7% -> k=3 needs 4 GDN state slots per seq; under final-track load admission is memory-capped again (same failure as E005, now at higher load). TPOT itself is excellent (23 fails vs 123 for E017).
Lesson: k=3 trades memory for TPOT; it only pays if state memory is freed elsewhere (fp8 KV, max-num-seqs bound, util). Val alone would have mis-ranked it — the final stress test caught it.
