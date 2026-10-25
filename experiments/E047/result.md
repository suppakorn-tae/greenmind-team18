# E047 result — no measurable long-context degradation from W8A8 -> re-quantization NOT justified
| server | exact (4/4 PINs) | PIN recall | ~9K | ~16K | ~22K | ~29K |
|---|---|---|---|---|---|---|
| BF16, configs/E030.sh | 0.8125 | 0.941 | 0.875 | 0.9375 | 0.75 | 0.6875 |
| W8A8, scripts/serve_final.sh (E046) | 0.8125 | 0.9375 | 0.9375 | 0.8125 | 0.75 | 0.75 |
Raw: `bf16_n64.json`, `w8a8_n64.json` (dry runs `dry*_w8a8.json`: plain lookup was 100% for both -> made harder with updates).
Conclusion: on long-range retrieval/tracking up to ~29K tokens W8A8 matches BF16 exactly. The official longctx 10/15 vs 13/15 (BF16, E039) is within binomial noise for n=15 (SE ~0.12), or is a different long-context skill (document comprehension) that this test doesn't cover. Keeping the 99.38 weights; a long-document QA comparison (open LongBench subsets) is the next check if the team wants more certainty.
