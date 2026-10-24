# E032 result — rejected
E030 + MTP k=3. val 20261024-200342-e032: 74/100, eta 1.59x, 241 J/req, fails ttft 19/tpot 8.
final 20261024-200457-e032: S 15.1, 100/300, p95 TTFT 8.0 s, fails ttft 221/tpot 77; peak 110 running / 28 waiting / KV 99.3%.
Even with the 3072 budget, k=3's 4th GDN state slot per sequence tips final peak into memory-capped admission. TPOT fails drop (122 -> 77) but TTFT explodes.
