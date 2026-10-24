# E033 result — CHAMPION (local), official pending
E030 + gpu-memory-utilization 0.96.
- val 20261024-200908-e033 / #2 20261024-201613-e033-2: 82 / 80 of 100, eta 1.77 / 1.74x, 256 / 251 J/req, 0 cutoffs (E030: 83/81, 1.77/1.72).
- final 20261024-201033-e033 / #2 20261024-201325-e033-2: S 30.2 / 34.4 (E030 28.8 / 25.5), 122 / 129 of 300 (123/114), TTFT fails 158 / 139 (170/179), p95 TTFT 3.13 / 2.86 s (4.23/4.57), peak waiting 4 (8-17). No OOM.
Final mean S 32.3 vs 27.2 (+19%); val tie. Extra KV relieves the final-peak admission queue -> better sT/sL, which is where the official headroom is (sT 0.71, sL 0.85).
- final #3 20261024-203549-e033-3: S 26.7, 124/300, TTFT fails 161, TPOT fails 118, p95 TTFT 3.05 s. E033 final x3: 30.2 / 34.4 / 26.7 (mean 30.4 vs E030 27.2).
- Frontend check during #3: API server instantaneous CPU max 25% (top, 1 s) -> not a bottleneck; --api-server-count not pursued.
- 20:39 paused for OFFICIAL final (user request).
- **OFFICIAL final (13:39 UTC): S 81.6** (E030 88.0), A 1.00, R 1.030, sE 0.90, sT 0.66, sL 0.76, eta 2.68x, 176 J/req, p95 TTFT 1.90 s, p95 TPOT 60 ms, SLO 231/300, acc 90.4%.
- Verdict: reverted to E030. Local edge +12% did not carry over (official -7%, likely within official noise).
