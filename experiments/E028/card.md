# E028 — champion E017 with MAX_BT 4096 -> 5120 (cap 1024)
Parent E017. One variable (rule-safe). Why: 3072 made final prefill slower -> KV-full queueing (E024); the opposite direction (5 cap-slots/step) may drain the peak prefill backlog faster and free memory sooner. Risk: longer prefill steps -> TPOT fails. Judge on val SLO + final. Start 2026-10-24T18:13:41+07:00.
