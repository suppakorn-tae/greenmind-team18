# E001 — AWQ-W4A16 + MTP k=3 (backlog #1)
- Parent: E000 (BF16 default baseline). Variable: switch to provided AWQ+MTP checkpoint with tuned flags (bundle = kit "awq-mtp" profile).
- Expect: lower J/request, better TPOT at peak (weights 8 GB vs 18 GB -> more KV, faster decode), fewer peak-phase SLO failures.
- Risk: accuracy loss (Thai), flag incompatibility on vLLM 0.32.0 (kit validated 0.24-0.27.1).
- Start: 2026-10-24 ~13:45 +07
