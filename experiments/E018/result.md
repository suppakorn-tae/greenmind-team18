# E018 result — rejected
val 20261024-154557-e018: 124/148 turns, 76/100 per arrival, eta 1.19x, 331 J/req, p95 TTFT 2.40 s, p95 TPOT 51 ms, fails ttft 14/tpot 8/cutoff 2, acc 79%.
vs E017 mean eta 1.37: -13% (beyond noise) -> no repeat/final. 512-token cap makes 20K prompts take ~40 steps (p95 TTFT 1.5 -> 2.4 s) and adds per-step overhead.
Lesson: the per-request prefill cap optimum is ~1024 on this trace (2048 too coarse, 512 too fine).
