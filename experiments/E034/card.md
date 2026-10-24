# E034 — champion E033 with gpu-memory-utilization 0.96 -> 0.98
Parent E033. One variable: +~0.8 GB more KV. Why: 0.92 -> 0.96 gave final mean S +19% by cutting peak admission queueing; follow the gradient. Risk: OOM at startup or under load (prefill activations, sampler); watch server log for OOM/errors and client errors. Start 2026-10-24T20:17:50+07:00.
