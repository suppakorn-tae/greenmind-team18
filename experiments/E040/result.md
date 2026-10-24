# E040 result — rejected
Short-first scheduler patch (cap 256 while prompts <= 1024 tok wait), verified active in API + EngineCore processes.
val 20261024-212859-e040: 80/100, eta 1.49x (E030 1.72-1.77), 269 J/req, fails ttft 14 / tpot 8 (E030 9-12 / 9), p95 TTFT 2.04 s (1.49-1.52).
final 20261024-213024-e040: S 29.2 (E030 28.8/25.5), 117/300, fails ttft 186 (170-179) / tpot 131, p95 TTFT 5.86 s (4.2-4.6).
Throttling in-flight long prefills while shorts wait pushes the TTFT misses onto long/RAG prompts (their 0.25 s/1K slack is used up when several overlap) and lengthens the energy window. The 1024 cap is already the balance point.
