# E017 result — CHAMPION (local), official pending
Config: E016 with --long-prefill-token-threshold 1024 (MAX_BT 4096), 400 W default.
- val #1 20261024-153804-e017 (fresh): 140/160 turns, 80/100 per arrival, eta 1.44x, 311 J/req, p95 TTFT 1.49 s, p95 TPOT 50 ms, fails ttft 10/tpot 9/cutoff 2, acc 80% (agent 4/20), local S 35.7 (A 0.74 agent noise).
- val #2 20261024-153948-e017-2: 138/160, 78/100, eta 1.30x, 338 J/req, fails ttft 12/tpot 9/cutoff 4, acc 76% (long 2/5, reason 11/15 -> A 0.34; noise: run #1 and final are normal).
- final 20261024-154141-e017: S 22.7 (E016 17.3, E008 15.5), A 1.00, eta 1.30x, 209 J/req, 117/300 per arrival, p95 TTFT 3.88 s, p95 TPOT 80 ms, fails ttft 175/tpot 123/cutoff 3, acc 81.0% (baseline 81.3%): agent 0.28 n=60, chat 1.0, long 0.6, rag 0.97, reason 0.93, thai 0.90.
vs E016: val TTFT fails 24-28 -> 10-12, per-arrival 63-68 -> 78-80, eta mean 1.215 -> 1.37; final S +31%.
Lesson: per-request prefill cap is the main TTFT lever on this trace; 1024 >> 2048.
- Confirmation (configs/champion.sh, fresh server, 400 W) 20261024-154907-e017-confirm: 140/160 turns, 80/100 per arrival, eta 1.34x, 334 J/req, p95 TTFT 1.48 s, p95 TPOT 50 ms, fails ttft 9/tpot 9/cutoff 3, acc 79% (agent 5/20), local val S 42.2 (A 1.00). Reproduces runs #1/#2.
- 15:51 paused for Tae's official run; champion left serving on :8000.
- **OFFICIAL final (2026-10-24 08:54 UTC): S 59.0, A 1.00, R 1.012, sE 0.53, sT 0.62, sL 0.80, eta 1.78x, 247 J/req, p95 TTFT 1.91 s, p95 TPOT 59 ms, SLO-met 215/300, accuracy 88.8%.** Linked to local final 20261024-154141-e017 (official #3).
- Calibration: local final is much harsher than official (117/300 vs 215/300, p95 TTFT 3.88 vs 1.91 s); local val tracks official val closely (E008 eta 1.22 vs 1.27). Use local val as primary, local final as a stress test.
- Local final variance (18:30): #2 20261024-183020-e017-final-2 S 22.1, 121/300, p95 TTFT 3.80 s, cutoff 3; #3 20261024-183314-e017-final-3 S 18.3, 115/300, p95 TTFT 3.99 s, cutoff 7. => champion local final S 22.7/22.1/18.3 (mean 21.0), never collapses; E024/E028 collapses are real.
