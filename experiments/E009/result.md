# E009 result — rejected
val 20261024-145308-e009: eta 0.92x, 381 J/req, 64/100, p95 TPOT 48 ms, A 1.0 (agent 5/20).
final 20261024-145509-e009: S 4.2, A 0.67 (agent 0.27 n=60), eta 1.04x, 244 J/req, 109/300, p95 TPOT 104 ms, fails tpot 134/ttft 169.
vs champion E008: J/req +23% val, final eta 1.04 vs 1.15. Fewer tokens/step costs more than the cheaper step saves.
Diagnosis (final load): peak 132 running/21 waiting, KV 96%; prefill ~9.7K tok/s while gen only 600-900 tok/s -> 8192-token prefill chunks (~0.85 s/step) starve decoders -> TPOT fails.
Lesson: keep k=2; the final-track TPOT tail is prefill interference, not spec-verify cost.
