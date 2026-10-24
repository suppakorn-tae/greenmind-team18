# E030 result — better than E027 (single run each track)
budget 3072. val 20261024-192201-e030: 143/160, 83/100, eta 1.77x, 258 J/req, p95 TTFT 1.49 s, p95 TPOT 50 ms, fails ttft 9/tpot 9/cutoff 0, acc 80% (reason 13/15).
final 20261024-192326-e030: S 28.8 (E027 mean 25.6), eta 1.46x, 195 J/req, 123/300, p95 TTFT 4.23 s, fails ttft 170/tpot 122, acc 81.0% (reason 0.956, chat 1.0, thai 0.90) — accuracy unchanged vs 5120.
- Replicate: val 20261024-193314-e030-2: 81/100, eta 1.72x, 261 J/req, cutoff 0; final 20261024-193434-e030-2: S 25.5, eta 1.39x, 196 J/req, 114/300, acc 80.7% (reason 0.933).
- E030 vs E027: val eta 1.77/1.72 (mean 1.75) vs 1.76/1.58/1.45/1.51 (mean 1.58) = +10%; final S 28.8/25.5 (mean 27.2) vs 25.2/25.0/26.6 (mean 25.6) = +6%; J/req val 258-261 vs 261-312; accuracy equal. -> CHAMPION (local), official pending.
