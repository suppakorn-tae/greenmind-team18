# E019 result — rejected
val 20261024-160538-e019: 115/146 turns, 70/100 per arrival, eta 1.11x, 332 J/req, p95 TTFT 1.45 s, p95 TPOT 50 ms, fails ttft 24/tpot 7/cutoff 3, acc 80%.
vs E017 (eta 1.30-1.44, 78-80/100, ttft fails 9-12): TTFT fails back up. With a 2048 budget and 1024 cap only ~2 prefills fit per step -> new arrivals queue again. TPOT improvement marginal (7 vs 9 fails).
Lesson: budget must leave several cap-sized slots (4096/1024 = 4); TPOT is not fixed by shrinking the budget.
