# E004 — ablation: E001 without MTP
Parent E001. One variable: remove --speculative-config (keep align mode, AWQ, all flags).
Hypotheses: (a) MTP k=3 per-step cost (~150 ms/step at ~50 seqs) causes peak TTFT/TPOT fails; (b) MTP multi-token deltas corrupt streaming qwen3_coder tool-call parsing -> agent 2/20.
Read: TTFT recovers + agent >=4/20 => MTP culprit; agent still 2/20 => AWQ culprit. Start 14:09.
