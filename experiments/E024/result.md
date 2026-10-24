# E024 result — rejected
val 20261024-172952-e024: 116/137, 79/100, eta 1.19x, 311 J/req, p95 TPOT 59 ms, fails ttft 10/tpot 9.
final 20261024-173131-e024: S 8.0 (E017 22.7), eta 1.05x, 215 J/req, 108/300, p95 TTFT 11.7 s, p95 TPOT 65 ms, fails ttft 212/tpot 66; peak 107 running / 31 waiting / KV 99.1%.
TPOT fails halved (123 -> 66) but slower prefill keeps requests resident longer -> KV full -> admission queueing -> TTFT explodes.
Lesson: on final-track load the binding constraint is KV/state memory at peak; budget 4096 is the floor.
