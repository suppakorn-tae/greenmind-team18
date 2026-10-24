# E037 — champion E030 + --quantization fp8 (online weight-only FP8 of the official BF16 weights)
Parent E030 (OFFICIAL 88.0). One variable.
Why: val power trace shows ~295 W even with 2 active requests in drain (drain = 36% of val energy): low-batch decode is HBM-bound on reading 18 GB BF16 weights per step. FP8 weights halve bytes/step (faster+cheaper decode at low/medium batch) and free ~9 GB for KV (peak memory cap; may re-enable k=3).
On Ampere vLLM uses FP8 Marlin (W8A16). Risk: Marlin slower at large batch (AWQ W4A16 was, E004); accuracy (8-bit per-channel usually near-lossless; check agent/thai/reason on val + final). Derived weights are made online at load from the official checkpoint (no new artifact). Start 2026-10-24T20:49:36+07:00.
