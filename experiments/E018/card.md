# E018 — champion E017 with long-prefill-token-threshold 1024 -> 512 (last new experiment, started before 16:15)
Parent E017. One variable. Hypothesis: follow the gradient (2048 -> 1024 cut TTFT fails 26 -> 11); 512 may remove the rest. Risk: 20K prompts need 40 steps -> long-prompt TTFT, per-step overhead, J/req. Val gate x2 then final. Start 2026-10-24T15:44:55+07:00.
