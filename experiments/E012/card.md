# E012 — champion E008 at 250 W power cap (backlog #4)
Parent E008. One variable: power limit 400 -> 250 W (set before server start, held for the whole run; no clock lock).
Hypothesis: avg draw 341 W; memory-bound decode loses little speed under a cap, while J/req drops ~15-25% -> eta up (s_E has weight 0.6). Risk: TPOT tail / prefill slower -> more tpot/ttft fails; if so try 300 W. Val gate first. Start 2026-10-24T15:10:08+07:00.
