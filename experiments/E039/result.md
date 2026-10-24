# E039 result — mechanism works, no score gain (kept as a safety option)
budget 1536 + reasoning_end_str transition phrase. val 20261024-210527-e039: 80/100, eta 1.79x, 248 J/req, 0 cutoffs, acc 81%. final 20261024-210646-e039: S 23.6, 116/300, 191 J/req (lowest), 0 cutoffs, acc 81.3% (reason 0.933, long 0.60).
The phrase fixes the answer-runaway failure of low budgets (E031/E038 cutoffs gone) with accuracy intact, but peak TPOT fails don't move (123): chat token volume isn't what drives peak step time.
Peak failure anatomy (val, E039): all 21 misses in peak; TTFT misses are short prompts at 0.52-0.88 s vs 0.51-0.65 s SLO; TPOT misses are short outputs (agent/rag/long, 51-72 ms) right after 5 long prompts (~65K prefill tokens) arrive within 3 s. vLLM priority policy only reorders the waiting queue / picks preemption victims (running partial prefills always go first), so EDF wouldn't help; v1 has no max_long_partial_prefills.
Async scheduling is already on (mtp is in EagleModelTypes).
