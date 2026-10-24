# E011 — champion E008 + --long-prefill-token-threshold 2048
Parent E008. One variable: per-request prefill chunk cap 2048 (only active when >1 request eligible; budget stays 8192).
Hypothesis: a long (18-20K) prompt no longer monopolizes the 8K budget, so short arrivals get scheduled in the next step -> removes staircase TTFT fails (~20 of 30); steps with long prefill get shorter -> TPOT tail too. Risk: long-prompt TTFT (SLO 0.5+0.25/1K = 5.5 s for 20K; with 2048/step at ~0.3 s steps ~3 s, OK). Val gate first. Start 2026-10-24T15:04:57+07:00.
