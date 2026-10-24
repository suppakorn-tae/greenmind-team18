# E033 result — CHAMPION (local), official pending
E030 + gpu-memory-utilization 0.96.
- val 20261024-200908-e033 / #2 20261024-201613-e033-2: 82 / 80 of 100, eta 1.77 / 1.74x, 256 / 251 J/req, 0 cutoffs (E030: 83/81, 1.77/1.72).
- final 20261024-201033-e033 / #2 20261024-201325-e033-2: S 30.2 / 34.4 (E030 28.8 / 25.5), 122 / 129 of 300 (123/114), TTFT fails 158 / 139 (170/179), p95 TTFT 3.13 / 2.86 s (4.23/4.57), peak waiting 4 (8-17). No OOM.
Final mean S 32.3 vs 27.2 (+19%); val tie. Extra KV relieves the final-peak admission queue -> better sT/sL, which is where the official headroom is (sT 0.71, sL 0.85).
