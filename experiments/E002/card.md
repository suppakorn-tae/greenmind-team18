# E002 — E001 + pre-measurement warmup script
- Parent: E001. Variable: run tools/warmup.py (random prompts, concurrency ramp 1..64, mixed lengths, tool calls) after server start, before the grader.
- Why: E001 fresh-server val had 51 TTFT fails; server log shows Triton JIT during inference (rejection/resample/GDN kernels); warm repeat gave 132/138 on-time, TPOT 20 ms, eta 1.85x.
- Expect: fresh+warmup val ≈ warm repeat. Gray area: warmup needs Tae's OK for official runs (done before measurement, constant state, random data, no response caching).
- Start: 2026-10-24 ~13:58
