# E026 result — tie with champion (not promoted)
cap 768. val 20261024-174820-e026 / #2 20261024-175332-e026-2: 83 / 79 of 100, eta 1.30x / 1.28x, 322 / 347 J/req, TTFT fails 6 / 9, p95_ttft_over_slo 0.98 / 1.03.
final 20261024-175000-e026 / #2 20261024-175520-e026-2: S 23.6 / 22.6 (E017 22.7), 116 / 116 of 300, eta 1.33x / 1.30x, p95 TTFT 4.4 / 4.6 s, fails ttft 175/180, tpot 125/124.
vs E017 (val 80/78/80, eta 1.44/1.30/1.34; final 22.7): SLO +1-2, eta -0.05, final S +0.4 -> all within noise. Keep officially-confirmed E017.
Lesson: cap optimum is flat between 768 and 1024.
## Re-evaluation 18:49 -> CHAMPION (local)
- val #3 20261024-184422-e026-3: 142/160, 82/100, eta 1.34x, 338 J/req, fails ttft 6/tpot 9/cutoff 4, p95_ttft_over_slo 0.97.
- final #3 20261024-184608-e026-3: S 25.3, eta 1.39x, 215 J/req, 118/300, fails ttft 176/tpot 122/cutoff 4, acc 81.7%.
- E026 x3 vs E017 x4: final S 23.6/22.6/25.3 (mean 23.8) vs 22.7/22.1/18.3/18.9 (mean 20.5) = +16%; final eta 1.30-1.39 vs 1.21-1.30; val SLO 83/79/82 vs 80/78/80/80; val TTFT fails 6/9/6 vs 9-12; val eta 1.31 vs 1.335 (tie). Accuracy equal.
Verdict: champion (local); official final pending (E017 official final 59.0).
