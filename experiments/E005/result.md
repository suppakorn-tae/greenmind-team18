# E005 result — promoted to mainline (not final-tested: TTFT still bad)
Run 20261024-141616-e005 val fresh: on-time 87/148, fails ttft 58/tpot 1/cutoff 2, p95 TTFT 4.4 s, p95 TPOT 36 ms, 298 J/req, acc 82% (agent 0.35, long 0.6, rag 0.95), A=1.0, eta 0.93x. MTP mean acceptance ~3.0 (k=3).
BF16 restores agent accuracy -> AWQ checkpoint confirmed as the agent-accuracy culprit (E001/E004 2/20).
TTFT diagnosis: peak KV usage 96.6% with 11 waiting at only 46 running; prefix-cache hit rate 3.3% over the whole run. KV pool = 321,877 tokens (vs 725K AWQ). Memory-bound admission => queueing TTFT.
Lesson: BF16+MTP is the accuracy-safe line; next free memory (prefix caching/align mode off -> E006).
