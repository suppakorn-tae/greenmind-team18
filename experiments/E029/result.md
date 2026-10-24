# E029 — abandoned (flag not applied; run = champion replicate)
The sed edit didn't match (E017 line is `--enable-chunked-prefill --no-enable-prefix-caching \`), so configs/E029.sh == E017. Runs are champion replicates:
val 20261024-183811-e029: 80/100, eta 1.26x, 354 J/req; final 20261024-184004-e029: S 18.9, 117/300, p95 TTFT 4.46 s, cutoff 3.
API-server CPU (ps lifetime avg) 35% — not obviously saturated; stream-interval deprioritised.
Champion E017 local stats now: val eta 1.44/1.30/1.34/1.26 (mean 1.335); final S 22.7/22.1/18.3/18.9 (mean 20.5).
