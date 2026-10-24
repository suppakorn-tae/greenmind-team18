# E010 result — rejected (within noise, not better)
val 20261024-150244-e010: S 18.2 (A 0.74, agent 4/20), eta 1.16x vs champion 1.22x, 301 J/req, 64/100, p95 TTFT 1.76 s, p95 TPOT 56 ms, fails ttft 28/tpot 9/cutoff 2.
Not beyond noise vs champion -> no final run (val gate).
TTFT diagnosis (champion val records): 30 late requests are mostly SHORT prompts (22-90 tok) in peak, with staircase TTFTs (2.6, 2.35, 2.1, 1.85 ... 0.56 s) after an 18-20K-token prompt arrives -> head-of-line blocking: a running long prefill takes the full chunk budget for ~3 steps, new short requests can't be scheduled.
Lesson: shrinking the budget for everyone doesn't help; cap the per-request chunk instead (E011).
