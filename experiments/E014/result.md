# E014 result — rejected
val 20261024-151757-e014 @300 W: 101/146, 56/100, eta 1.19x, 271 J/req, avg 294 W, p95 TTFT 2.15 s, p95 TPOT 71 ms, fails ttft 37/tpot 10.
val #2 20261024-151944-e014-2: 100/148, 53/100, eta 1.04x, 308 J/req, fails ttft 37/tpot 10.
Mean eta 1.11x < champion 1.22x; SLO 53-56/100 vs 63. Power capping trades J for TTFT fails roughly 1:1 at this load (prefill is compute-bound).
Lesson: no power cap on this workload at 400 W-class load; stay at default.
