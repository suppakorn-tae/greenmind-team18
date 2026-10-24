# E046 — E045 with served model name "mango" (label only)
Identical to E045 (weights suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8 @ 1909e853, team W8A8 INT8 derivative of CMKL/MANGO1.5-Qwen3.5-9B). Only the API label (/v1/models id) changes, at the team lead's request for the official run. No effect on serving or measurement. Weights remain disclosed in CHAMPION.md / README / HF link for final deliverables. 2026-10-24T23:39:02+07:00

## Result
local final 20261024-234133-e046-final ("mango" label, KV pool 608,632 tokens): **S 94.8**, A 1.00, eta 3.11x, 179 J/req, 257/300 per arrival (389/438 turns), p95 TTFT 1.63 s, p95 TPOT 49.6 ms, fails ttft 29 / tpot 21 / cutoff 0; peak 91 running / 0 waiting / KV 54%; acc 82.3% (agent .38, chat 1.0, long .87, rag .90, reason .87, thai .93). Server log clean. Consistent with E045 (93.7 / 95.0).

## OFFICIAL final (2026-10-24 16:48 UTC): **S 99.38**
A 1.00, R 0.987, sE 1.00, sT 0.969, sL 1.00 (ttft 1.0, tpot 1.0), eta 3.82x, 157 J/req, n_ok 294/300, 0 truncated, 0 errors, p50 TTFT 0.26 s, p95 TTFT 1.21 s (0.81x SLO), p95 TPOT 37 ms, accuracy 88.3% micro.
Class accuracy / retention: chat .856/.96, rag .95/1.04, longctx .667/**0.71**, reasoning .911/1.03, agentic .85/1.06, thai .967/1.16. gate_reasons [] — longctx is below 0.75 retention but has n=15 (< 20, not gated). Watch: if a task set had >= 20 long-context items this would cut A.
Full JSON in notes/.
