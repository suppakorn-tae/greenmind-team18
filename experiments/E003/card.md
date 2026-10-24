# E003 — max-num-batched-tokens 8192 -> 2048 (backlog #6)
Parent E002. One variable: MAX_BT 2048. Hypothesis: shorter engine steps remove the peak-phase TTFT stalls for short prompts (chat/thai/reason/agent), fixing ~50 TTFT fails. Risk: long-prompt (20K) TTFT, prefill efficiency. Start 14:01.
