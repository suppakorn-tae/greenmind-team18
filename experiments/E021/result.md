# E021 result — rejected
val 20261024-165424-e021: 119/148, 72/100, eta 1.13x, 337 J/req, p95 TTFT 2.04 s, p95 TPOT 60 ms, fails ttft 17/tpot 10/cutoff 3, acc 77% (agent 2/20).
fp8 KV: pool 475K -> 702K tokens, but on Ampere it forces FLASHINFER (instead of FLASH_ATTN) and "FULL_AND_PIECEWISE CUDA graphs not supported with spec-decode for FlashInfer" -> slower decode steps. eta -17% vs champion mean (1.36), beyond noise; agent 2/20 also a warning.
Lesson: fp8 KV is a net loss on A100 with MTP (backend/cudagraph downgrade outweighs memory).
