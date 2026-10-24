# E013 result — rejected
val 20261024-151438-e013 @250 W + chunk cap 2048: on-time 90/148, 45/100, eta 1.08x, 267 J/req, p95 TTFT 2.25 s, p95 TPOT 47 ms, fails ttft 50/tpot 7, A 1.0.
Chunk cap gives no TTFT relief under 250 W -> at 250 W the GPU is simply too slow for peak load (not just head-of-line blocking).
Lesson: 250 W is past the knee; try 300 W.
