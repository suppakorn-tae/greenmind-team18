# E005 — BF16 + MTP k=3 + tuned flags
Parent E001 (swap AWQ -> BF16 weights; everything else identical). Hypothesis: BF16 restores agent accuracy (>=4/20) and BF16 decode at batch is faster than Marlin W4A16, while MTP cuts J/req. Risk: less KV memory (18 GB weights). If agent still 2/20 -> tuned flags (language-model-only/align/prefix) are the culprit. Start 14:14.
