# E013 — E012 (250 W) + --long-prefill-token-threshold 2048
Parent E012. One variable vs parent: per-request prefill chunk cap 2048 (E011's TTFT fix). Hypothesis: keep the -18% J/req of 250 W while removing the head-of-line TTFT staircase that the cap made worse -> eta > champion. Val gate. Start 2026-10-24T15:13:34+07:00.
