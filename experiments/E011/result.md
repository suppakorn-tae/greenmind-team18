# E011 result — rejected on eta (but best s_L / TTFT; candidate to recombine later)
val #1 20261024-150609-e011: 100/136 turns, 66/100 per arrival, eta 1.01x, 315 J/req, p95 TTFT 1.50 s, p95 TPOT 67 ms, fails ttft 23/tpot 10, s_L 0.74.
val #2 20261024-150807-e011-2: 110/148, 64/100, eta 1.10x, 320 J/req, p95 TTFT 1.32 s, p95 TPOT 60 ms, fails ttft 26/tpot 9.
Mean eta ~1.05x vs champion 1.22x (single run). TTFT tail clearly better (p95_ttft_over_slo 2.7 -> 1.36, s_L 0.56 -> 0.74) but on-time count/energy not better; turn count varies 136-158 run to run (agent episodes), which adds noise to eta.
Score math: s_E (log, weight 0.6) dominates at eta ~1.1-1.3 (1.22 -> 0.18, 1.05 -> 0.04); s_L gain (+5.7% factor) doesn't pay for it.
Lesson: chunk cap fixes the TTFT staircase but not η; energy is the bigger lever now.
