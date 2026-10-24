# E023 result — not champion locally (mixed); candidate for an official A/B
Config: E017 + SM clock locked 1215 MHz (`greenmind-kit/power.sh 400 1215`), held for the whole run.
- val 20261024-170905-e023: 114/148, 67/100, eta 1.39x, 261 J/req, avg 263 W, p95 TTFT 2.17 s, p95 TPOT 65 ms, fails ttft 23/tpot 11/cutoff 3, A 1.00, S 41.2.
- val #2 20261024-171103-e023-2: 119/148, 72/100, eta 1.43x, 265 J/req, avg 266 W, fails ttft 17/tpot 11, A 1.00, S 44.6.
- final 20261024-171302-e023: S 20.2 (champion E017 22.7), A 1.00, eta 1.37x (1.30), 176 J/req (209), 98/300 (117), p95 TTFT 6.35 s (3.88), p95 TPOT 90 ms (80), fails ttft 204/tpot 148.
vs E017: energy -16..-21% on both tracks and eta higher on both, but SLO-met lower (val 67-72 vs 78-80; final 98 vs 117) -> s_T/s_L lose what s_E gains on the harsh local final.
Official final is much lighter than local final (E017: 215/300 official vs 117/300 local), so the TTFT penalty may be smaller officially while the -20% J/req carries over -> worth one official A/B vs E017 (official S 59.0).
- **OFFICIAL final (10:17 UTC): S 52.5** (E017: 59.0), A 1.00, R 1.015, sE 0.54 (0.53), sT 0.37 (0.62), sL 0.69 (0.80), eta 1.80x (1.78x), 186 J/req (247), p95 TTFT 2.15 s (1.91), p95 TPOT 74 ms (59), SLO 164/300 (215), acc 89.1%.
- Verdict: rejected. Energy saved is cancelled by lost on-time requests (eta flat) and sT/sL drop. Local final ranked it correctly (20.2 < 22.7) -> trust local final over val for power/clock changes.
