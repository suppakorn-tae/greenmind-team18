# E046 result — FINAL SUBMISSION (OFFICIAL Final S 99.38)
= E045 (team W8A8 INT8 weights + MTP k=2 + bf16 SSM state + prefill 4096/1024 + thinking budget 3072) with `--served-model-name mango`.
- local final `20261024-234133-e046-final`: S 94.8, A 1.00, eta 3.11x, 179 J/req, 257/300, p95 TTFT 1.63 s, p95 TPOT 49.6 ms, 0 cutoffs, accuracy 82.3%.
- **OFFICIAL Final attempt `20261024T164844-c109c6` (2026-10-24 16:48 UTC): S 99.38**, A 1.00, R 0.987, sE 1.00, sT 0.969, sL 1.00, eta 3.82x, 157.5 J/req, 294/300 on time, 0 truncated / 0 errors, p95 TTFT 1.21 s, p95 TPOT 37 ms, accuracy 88.3%. Full export: `notes/official_20261024T164844.json` and `results/official/`.
- Weak spot: longctx retention 0.71 (15 items; below the 20-item gate size, so not gated).
See `card.md` for details.
