# E002 result — rejected (warmup didn't help)
Run 20261024-1355 (val, fresh server + tools/warmup.py): on-time 81/137, fails ttft 52 / tpot 10 / cutoff 2, p95 TTFT 3.2 s, p95 TPOT 58 ms, 269 J/req, acc 78% (agent 0.10), eta 0.96x.
Diagnosis: E001#2 'warm' gain (132/138) was a prefix-cache artifact (identical trace replayed), NOT JIT. Late TTFT is peak-phase and hits even 26-token chat prompts (TTFT up to 3.2 s) => engine steps stalled by 8192-token prefill chunks (AWQ prefill ~5-7K tok/s => ~1.2 s/step).
Lesson: always evaluate on a fresh server (repeats on one server are inflated by prefix cache). Next: smaller max-num-batched-tokens (E003).
