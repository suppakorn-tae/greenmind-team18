# E028 result — rejected
MAX_BT 5120 / cap 1024. val 20261024-182357-e028: 116/136, 80/100, eta 1.16x, 318 J/req, p95 TPOT 62 ms, fails ttft 9/tpot 9/cutoff 3.
final 20261024-182542-e028: S 0.0, eta 0.92x, 99/300, p95 TTFT 13.1 s, fails ttft 218/tpot 74; peak 102 running / 36 waiting / KV 97%.
Both 3072 (E024) and 5120 collapse into KV-full backlogs on local final while 4096 doesn't -> suspect local final is bistable (backlog forms or not). Champion final variance check next.
