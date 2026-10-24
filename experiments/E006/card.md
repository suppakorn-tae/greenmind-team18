# E006 — E005 with prefix caching OFF (and therefore no mamba align mode) (backlog #7)
Parent E005. One variable: --no-enable-prefix-caching (align mode only exists for prefix caching).
Hypothesis: align-mode caching of per-block GDN states eats the KV pool (96.6% at 46 seqs) while giving 3% hit rate; turning it off raises concurrency headroom and removes peak admission queueing -> fewer TTFT fails. Risk: agentic multi-turn re-prefill cost (small prompts, minor). Start 14:22.
