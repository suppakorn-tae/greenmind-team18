# E025 result — rejected (tie)
util 0.96, KV pool 518,705 tokens (E017 475,367). No OOM.
val 20261024-173556-e025 / #2 20261024-174201-e025-2: eta 1.20x / 1.28x, 77 / 77 of 100, 360 / 334 J/req.
final 20261024-173751-e025 / #2 20261024-174344-e025-2: S 22.9 / 22.3 (E017 22.7), 124 / 116 of 300, p95 TTFT 2.9 / 3.0 s (3.88), TTFT fails 152/167 (175), TPOT fails 126/128 (123).
(Runs 20261024-1741xx-e025-2 with idle 185 W were an invalid concurrent val+final launch, interrupted; ignore.)
Lesson: +9% KV pool trims final TTFT tail slightly but not S; val eta lower. Not worth the OOM risk.
