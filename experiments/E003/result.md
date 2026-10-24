# E003 result — rejected
Val fresh, no warmup, MAX_BT 2048: on-time 72/135, fails ttft 57/tpot 14/cutoff 4, p95 TTFT 5.6 s, p95 TPOT 55 ms, 283 J/req, acc 76% (agent 0.10), eta 0.81x.
Worse than E002 (8192) => smaller chunks add per-step fixed cost; stalls are not from big prefill chunks. Suspect MTP/per-step cost (~150 ms/step at ~50 seqs) -> ablation E004 (no MTP).
