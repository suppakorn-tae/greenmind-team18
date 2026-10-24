# E008 result — CHAMPION
Config: BF16 + MTP k=2 + SSM state bf16, no prefix caching, language-model-only, len 41984, MAX_BT 8192, 400 W default.
- OFFICIAL val (2026-10-24 07:37 UTC): S 31.5, A 1.00, R 1.019, sE 0.22, sT 0.51, sL 0.59, eta 1.27x, 333 J/req, p95 TTFT 2.02 s, p95 TPOT 38 ms, SLO 65/100, accuracy 88.9% (baseline 87.2%).
- Local val 20261024-144413-e008 (fresh): S 21.2 (A 0.74 from noisy agent 4/20), eta 1.22x, 311 J/req, 63/100, p95 TTFT 1.69 s, p95 TPOT 59 ms, fails ttft 30/tpot 9/cutoff 2.
- Local final 20261024-144601-e008 (same server after val; no prefix cache so no warm artifact): S 15.5, A 1.00, eta 1.15x, 224 J/req, 119/300, p95 TTFT 3.66 s, p95 TPOT 88 ms, fails ttft 163/tpot 110/cutoff 7; agent 0.33 n=60.
- MTP per-position acceptance 0.87, 0.74. Peak running 56, KV 48%.
Calibration: local val eta 1.22 vs official 1.27 -> agrees. Local val A is pessimistic (agent n=20 noise); trust eta + final-track A.
Lesson: k=2 >= k=3 (eta 1.22 vs 1.12, within noise); on final track the bottleneck is per-step compute (TPOT tail) at high concurrency.
- Champion E008 local val repeat: eta 1.22 / 1.13 (mean 1.18) -> local run-to-run spread ≈ ±5-8% on eta; agent turn count varies 136-158 per run.
