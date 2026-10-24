# E006 result — neutral (kept as new parent for simplicity)
Run 20261024-142219-e006 val fresh: on-time 86/148 (49/100 per arrival), fails ttft 59/tpot 1/cutoff 3, p95 TTFT 4.0 s, p95 TPOT 33 ms, 328 J/req, acc 76% (agent 0.15 = 3/20), eta 0.84x. KV pool 332,606 tokens; peak still 98.3% KV with 12 waiting at 46 running.
vs E005: within noise on SLO; prefix caching gave no TTFT benefit (hit rate 3%) — but also not the memory hog.
Agent class on BF16 spans 3-7/20 across runs (baseline 4-6) => agent n=20 is very noisy; AWQ is a consistent 2/20.
Baseline (fresh) has only 10 TTFT fails vs ~58 for BF16+MTP => new suspect: MTP multiplies per-seq GDN state (k+1 state slots) -> ~46 seq memory cap.
