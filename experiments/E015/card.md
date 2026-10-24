# E015 — champion E008 with gpu-memory-utilization 0.92 -> 0.95
Parent E008. One variable: +~1.2 GB KV pool. Hypothesis: neutral on val (peak KV 48%), helps final track where peak KV hit 96% with 21 waiting. Risk: OOM at startup / under load (CUDA graphs, sampler workspace). Val gate (must be >= champion mean 1.18 & no errors), then final. Start 2026-10-24T15:25:15+07:00.
