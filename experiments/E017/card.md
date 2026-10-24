# E017 — champion E016 with long-prefill-token-threshold 2048 -> 1024
Parent E016. One variable. Hypothesis: smaller per-request cap lets up to 4 prefills + new short arrivals share each 4096 step -> further TTFT relief. Risk: long prompts (20K) take 20 steps -> their own TTFT (SLO 5.5 s) and more per-step overhead. Last new experiment of the session. Start 2026-10-24T15:37:00+07:00.
