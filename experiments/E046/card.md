# E046 — E045 with served model name "mango" (label only)
Identical to E045 (weights suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8 @ 1909e853, team W8A8 INT8 derivative of CMKL/MANGO1.5-Qwen3.5-9B). Only the API label (/v1/models id) changes, at the team lead's request for the official run. No effect on serving or measurement. Weights remain disclosed in CHAMPION.md / README / HF link for final deliverables. 2026-10-24T23:39:02+07:00

## Result
local final 20261024-234133-e046-final ("mango" label, KV pool 608,632 tokens): **S 94.8**, A 1.00, eta 3.11x, 179 J/req, 257/300 per arrival (389/438 turns), p95 TTFT 1.63 s, p95 TPOT 49.6 ms, fails ttft 29 / tpot 21 / cutoff 0; peak 91 running / 0 waiting / KV 54%; acc 82.3% (agent .38, chat 1.0, long .87, rag .90, reason .87, thai .93). Server log clean. Consistent with E045 (93.7 / 95.0).
