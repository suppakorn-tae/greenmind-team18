# E030 result — better than E027 (single run each track)
budget 3072. val 20261024-192201-e030: 143/160, 83/100, eta 1.77x, 258 J/req, p95 TTFT 1.49 s, p95 TPOT 50 ms, fails ttft 9/tpot 9/cutoff 0, acc 80% (reason 13/15).
final 20261024-192326-e030: S 28.8 (E027 mean 25.6), eta 1.46x, 195 J/req, 123/300, p95 TTFT 4.23 s, fails ttft 170/tpot 122, acc 81.0% (reason 0.956, chat 1.0, thai 0.90) — accuracy unchanged vs 5120.
- Replicate: val 20261024-193314-e030-2: 81/100, eta 1.72x, 261 J/req, cutoff 0; final 20261024-193434-e030-2: S 25.5, eta 1.39x, 196 J/req, 114/300, acc 80.7% (reason 0.933).
- E030 vs E027: val eta 1.77/1.72 (mean 1.75) vs 1.76/1.58/1.45/1.51 (mean 1.58) = +10%; final S 28.8/25.5 (mean 27.2) vs 25.2/25.0/26.6 (mean 25.6) = +6%; J/req val 258-261 vs 261-312; accuracy equal. -> CHAMPION (local), official pending.
- **OFFICIAL final (12:57 UTC): S 88.0** (E017 59.0), A 1.00, R 1.034, sE 0.96, sT 0.71, sL 0.85, eta 2.87x, 173 J/req, p95 TTFT 1.85 s, p95 TPOT 57 ms, SLO 242/300, accuracy 90.7%.
- Official gain (+49%) far larger than local (+30% vs E017): official trace evidently has more long-thinking requests hitting the cutoff.
- With sE 0.96 (cap at 3x), further gains must come from sT/sL: trade energy for latency now.
