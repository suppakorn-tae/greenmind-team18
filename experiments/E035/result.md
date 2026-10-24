# E035 result — rejected
MAX_BT 3072 on E033. val 20261024-202222-e035: 83/100, eta 1.33x, 287 J/req, p95 TPOT 61 ms. final 20261024-202356-e035: S 28.7 (E033 30.2/34.4), 123/300, TTFT fails 167, TPOT fails 120 (same), peak 101 running / 11 waiting / KV 77% (no collapse now).
Lesson: final-peak TPOT fails (~120) are insensitive to prefill chunking -> GPU decode-saturated at ~140 concurrent sequences.
