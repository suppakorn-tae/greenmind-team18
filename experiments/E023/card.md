# E023 — champion E017 with SM clock locked at 1215 MHz (400 W limit)
Parent E017. One variable: lock SM clock 1215 MHz (max 1410), constant for the whole run.
Why: ~50% of val energy is the drain phase (1-3 long reasoning decodes, memory-bound, GPU still at boost clocks). Lower SM clock cuts power at all load levels; memory-bound decode speed nearly unchanged. Risk: compute-bound peak prefill ~14% slower -> TTFT fails (prefill cap should absorb some). Val gate x2. Start 2026-10-24T17:01:49+07:00.
