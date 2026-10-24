# Experiments

| id | parent | change | status | val S | final S | official S | J/req | SLO | acc | lesson |
|---|---|---|---|---|---|---|---|---|---|---|
| E000 |  | default vllm serve BF16 baseline | baseline | 0.0 | 0.0 | 0.0 | 467 | 64/100 | 87.2 | official val baseline: s_T 0.51 s_L 0.59 p95 TTFT 2.06s |
| E001 | E000 | AWQ+MTP k=3 tuned flags | rejected-cold | 0.0 |  |  | 272 | 45/100 cold; 132/138 warm | 77.0 | cold: 51 TTFT fails from Triton JIT in inference. warm: eta 1.85x, TPOT 20ms, 227 J/req. agent acc 0.1 vs baseline 0.2-0.3 (n=20, A=0.05) - suspect AWQ |
| E002 | E001 | warmup.py | rejected | 0.0 |  |  | 269 | 81/137 | 78.0 | warm repeats are prefix-cache artifacts; peak TTFT stalls from 8192-token prefill chunks |
| E003 | E002 | max-num-batched-tokens 2048 | rejected | 0.0 |  |  | 283 | 72/135 | 76.0 | smaller chunks worse; per-step fixed cost dominates, suspect MTP; ablate |
| E004 | E001 | remove MTP (ablation) | rejected | 0.0 |  |  | 315 | 80/134 | 77.0 | agent 2/20 without MTP too -> AWQ (or flags) culprit; MTP improves J/req |
| E005 | E001 | AWQ -> BF16 (MTP k=3 + flags) | promoted-mainline | 0.0 |  |  | 298 | 87/148 | 82.0 | BF16 restores agent 7/20 (A=1.0); TTFT fails from KV-full admission queueing at peak (96.6% KV, prefix hit 3%) |
| E006 | E005 | prefix caching off | neutral | 0.0 |  |  | 328 | 86/148 | 76.0 | prefix cache irrelevant (3% hits); KV still full at 46 seqs; agent class noisy 3-7/20 on BF16 |
| E007 | E006 | mamba ssm cache bf16 | promoted | 14.5 |  |  | 303 | 106/148 (60/100) | 79.0 | bf16 SSM state frees memory: peak KV 98->56%, TTFT fails 59->33; now TPOT-bound (60ms) |
| E008 | E007 | champion repeat | champion | 21.2 | 15.5 | 31.5 | 302 | 63/100 | 79.0 | champion val eta over 2 fresh runs: 1.22, 1.13 (mean 1.18, spread +-0.05) |
| E009 | E008 | MTP k 2->1 | rejected | 0.0 | 4.2 |  | 381 | 64/100 | 80.0 | k=1 worse J/req and final TPOT; final TPOT tail = 8K prefill chunks starving decode |
| E010 | E008 | max-num-batched-tokens 4096 | rejected | 18.2 |  |  | 301 | 64/100 | 80.0 | within noise; TTFT fails are head-of-line blocking of short prompts behind long prefills |
| E011 | E008 | long-prefill-token-threshold 2048 | rejected | 14.1 |  |  | 318 | 66/100, 64/100 | 79.0 | s_L 0.56->0.74 but eta mean 1.05 vs 1.22; s_E dominates |
| E012 | E008 | power cap 250 W | rejected | 19.5 |  |  | 254 | 45/100 | 82.0 | 250W: J/req -18% but TTFT fails 30->48 (prefill slower) |
| E013 | E012 | 250W + chunk cap 2048 | rejected | 14.3 |  |  | 267 | 45/100 |  | 250W too slow at peak regardless of chunking |
| E014 | E008 | power cap 300 W | rejected | 1.2 |  |  | 290 | 56/100, 53/100 |  | 300W: J/req -7..-13% but TTFT fails up; eta mean 1.11 < 1.22 |
| E015 | E008 | gpu-memory-utilization 0.95 | rejected | 0.0 |  |  | 344 | 62/100 |  | no val gain; more energy |
| E016 | E010 | MAX_BT 4096 + long-prefill-threshold 2048 | champion | 30.8 | 17.3 |  | 319 | 68/100, 63/100 | 81.0 | chunk 4096 + per-request cap 2048 fixes TTFT staircase; final S 17.3 > 15.5 |
| E017 | E016 | local final variance | champion | 42.2 | 21.0 | 59.0 | 247 | 215/300 | 88.8 | champion final S 22.7/22.1/18.3; cutoffs 3-7 per run drive variance |
| E018 | E017 | long-prefill-threshold 512 | rejected | 22.5 |  |  | 331 | 76/100 | 79.0 | 512 too fine: long-prompt TTFT up; optimum ~1024 |
| E019 | E017 | MAX_BT 2048 (cap 1024) | rejected | 20.6 |  |  | 332 | 70/100 | 80.0 | budget 2048 with cap 1024 re-creates queueing; keep 4096/1024 |
| E020 | E017 | MTP k 2->3 on E017 | rejected | 31.7 | 2.9 |  | 292 | 75/100, 75/100 | 79.0 | val tie but final collapses: k+1 GDN state slots -> KV 98.7%, 51 waiting, p95 TTFT 19.8 s |
| E021 | E017 | kv-cache-dtype fp8 | rejected | 1.2 |  |  | 337 | 72/100 | 77.0 | fp8 KV forces FlashInfer + downgraded cudagraphs w/ spec decode -> TPOT 60ms |
| E022 | E017 | power cap 350 W | rejected | 30.2 |  |  | 310 | 77/100, 70/100 | 79.0 | 350W slightly worse; ~50% of val energy is drain phase (cutoff reasoning reqs at low batch) |
| E023 | E017 | SM clock lock 1215 MHz | official-final (worse than champion) | 44.6 | 20.2 | 52.5 | 186 | 164/300 | 89.1 | OFFICIAL final S 52.5 < E017 59.0: J/req -25% but on-time 215->164, sT 0.62->0.37, sL 0.80->0.69; eta only 1.78->1.80. Local final ranking was correct |
| E024 | E017 | MAX_BT 3072 | rejected | 29.9 | 8.0 |  | 311 | 79/100; 108/300 | 79.0 | TPOT fails halve on final but KV-full queueing -> TTFT 11.7s |
| E025 | E017 | gpu util 0.96 | rejected | 37.0 | 22.9 |  | 347 | 77/100; 124/300 | 79.0 | final tie (22.9/22.3 vs 22.7), val eta lower |
| E026 | E017 | long-prefill-threshold 1024->768 | official-final (worse than E017) | 40.5 | 23.8 | 46.6 | 264 | 193/300 | 90.3 | OFFICIAL final 46.6 < E017 59.0 despite local final +16% -> local/official disagree on a 1-flag change; official noise likely large; E017 stays the base |
| E027 | E017 | E017 + server-side thinking budget 5120 (user-authorized) | paused; serving configs/champion.sh (=E027) for official final | 51.1 | 26.6 |  | 297 | 83/100; 120/300 | 81.0 | budget removes 8192 cutoffs (always failures) and the drain-phase energy; final mean +25% vs E017 |
| E028 | E017 | MAX_BT 5120 | rejected | 27.7 | 0.0 |  | 318 | 80/100; 99/300 | 80.0 | final backlog collapse (36 waiting); local final may be bistable |
| E029 | E017 | stream-interval 4 (not applied) | abandoned |  |  |  |  |  |  | config edit failed -> champion replicate: val eta 1.26, final S 18.9 |
| E030 | E027 | thinking budget 5120->3072 | official-final CHAMPION | 47.4 | 27.2 | 88.0 | 173 | 242/300 | 90.7 | OFFICIAL final S 88.0 (E017 59.0): sE 0.96, sT 0.71, sL 0.85, eta 2.87x, 173 J/req, 242/300, p95 TTFT 1.85 s, acc 90.7%. sE near cap -> next gains must come from sT/sL |
| E031 | E030 | thinking budget 2048 | rejected | 50.5 | 26.2 |  | 305 | 81/100; 115/300 | 81.0 | 2048 too tight: answer can then run to cutoff; 3072 best |
| E032 | E030 | MTP k=3 with budget | rejected | 53.4 | 15.1 |  | 241 | 74/100; 100/300 | 81.0 | k=3 still memory-collapses on final even with budget |
| E033 | E030 | final replicate + frontend CPU check | official-final (worse than E030) -> reverted | 63.4 | 30.4 | 81.6 | 176 | 231/300 | 90.4 | OFFICIAL final 81.6 < E030 88.0 (sT 0.66 vs 0.71, sL 0.76 vs 0.85) despite local +12%; second time a <=16% local edge fails officially |
| E034 | E033 | gpu util 0.98 | crashed |  |  |  |  |  |  | CUDA OOM under load; 0.96 is the ceiling |
| E035 | E033 | MAX_BT 3072 | rejected | 38.7 | 28.7 |  | 287 | 83/100; 123/300 | 78.0 | TPOT fails insensitive to chunking; decode-saturated |
| E036 | E033 | max-num-seqs 112 | rejected | 45.7 | 19.0 |  | 240 | 83/100; 111/300 | 79.0 | admission cap: TPOT fails down, TTFT fails up more |
| E037 | E030 | online fp8 weights | rejected | 0.0 | 0.0 |  | 254 | 47/100; 61/300 | 80.0 | FP8 Marlin W8A16 slow at batch on A100 + accuracy dip |
| E038 | E030 | thinking budget 1536 | rejected | 63.8 | 17.5 |  | 248 | 81/100; 112/300 | 81.0 | low budgets -> answer runaway to 8192 cutoff (reason class) |
| E039 | E038 | forced-end transition phrase (budget 1536) | neutral (safety option) | 62.8 | 23.6 |  | 248 | 80/100; 116/300 | 81.0 | phrase removes low-budget answer runaway; no SLO gain; peak is prefill-capacity bound |

## Key findings

- AWQ-W4A16 checkpoint: agent class deterministic 2/20 (BF16: 3-7/20) and slower TPOT under batch (Marlin) -> rejected; BF16 is the line.
- MTP on GDN hybrid multiplies per-sequence recurrent state (k+1 slots); with fp32 state this capped concurrency at ~46 seqs (KV 98%, queueing TTFT). `--mamba-ssm-cache-dtype bfloat16` fixed it (peak KV 56%, TTFT fails 59->33), no visible accuracy loss.
- MTP k=2 >= k=3 (acceptance 0.87/0.74/0.59 per position).
- Prefix caching: 3% hit rate on this workload; align mode no benefit -> off.
- Smaller prefill chunks (2048) hurt: per-step fixed cost.
- Repeat runs on one server with prefix caching are inflated (warm cache replays) — always fresh server, or prefix caching off.
- Local val eta tracks official (1.22 vs 1.27); local val A is pessimistic because the agent class (n=20) is noisy.
- Power cap 250/300 W: J/req -13..-18% but prefill (compute-bound) slows -> TTFT fails up; eta not better. Stay at 400 W default.
- MTP k=1 worse than k=2 (J/req +23%).
- TTFT fails at peak are a head-of-line staircase: short prompts wait behind a long (18-20K) prompt's prefill chunks. MAX_BT 4096 + --long-prefill-token-threshold 2048 together fix it (p95 TTFT 1.7 -> 1.3 s, final S 15.5 -> 17.3); each alone was neutral.
- gpu-memory-utilization 0.95: no val benefit.
- Noise: eta counts turns; agent episodes vary 136-158 turns/run. Per-arrival on-time per kJ is a steadier secondary metric. Champion repeat spread ±5-8% eta.
- Per-request prefill cap 1024 with a 4096 budget (E017) is the big TTFT win: val TTFT fails ~30 -> ~10, per-arrival 63 -> 80/100. 2048 and 512 caps worse; budget 2048 worse (E019).
- OFFICIAL: E008 val S 31.5; E017 final S 59.0 (eta 1.78x, 247 J/req, 215/300, acc 88.8%).
- Local final track is harsher than official (117/300 vs 215/300) but ranks configs the same; local val tracks official val.
- MTP k=3 on the champion (E020): val tie with -10% J/req, but final-track collapse (memory-capped admission, p95 TTFT 19.8 s). Always confirm on local final; val can't see this failure.
- Clock lock 1215 MHz (E023): OFFICIAL final 52.5 vs 59.0 — J/req -25% but on-time -24%; eta flat, sT/sL down. On this workload, any clock/power reduction loses SLO about 1:1 with energy. Local final predicted the ranking; val didn't.
- [GRAY, local only] Server-side thinking budget 5120 (E027): val eta 1.36 -> 1.67, J/req -17%, final S 22.7 -> 25.2, zero cutoffs, accuracy equal or better. Biggest remaining lever; needs organizer approval.
- Cap 768 (E026): local final +16% over E017 (3 vs 4 runs) but OFFICIAL final 46.6 vs 59.0 -> a 1-flag local edge of ~15% is not reliable officially; official run-to-run noise is probably ±15-20%.
- **Thinking budget 3072 (E030): OFFICIAL final S 88.0** (from 59.0): eta 1.78 -> 2.87x, J/req 247 -> 173, SLO 215 -> 242/300, accuracy 88.8 -> 90.7%. sE now 0.96 (near the 3x cap) -> remaining headroom is sT 0.71 / sL 0.85 (latency).
- Official calibration: local-final edges of +12% (E033) and +16% (E026) both LOST officially (81.6 vs 88.0; 46.6 vs 59.0). Official run-to-run noise is probably ±10%; only structural changes (thinking budget: local +30%, official +49%) carried over. Don't spend official runs on <20% local edges.
- Quantization on A100 (AWQ W4, online FP8 W8): Marlin kernels slow at serving batch; FP8 also costs accuracy (E037). Dead end.
- Thinking budget below ~3072 makes some reason answers run away to the 8192 cutoff; a transition phrase in `--reasoning-config reasoning_end_str` fixes that (E039) but lower budgets don't improve peak SLO.
- Remaining loss is peak prefill capacity: ~65K long-prompt tokens arriving within 3 s at peak; misses are marginal (TTFT 0.52-0.88 vs 0.51-0.65 s; TPOT 51-72 ms). Score sensitivity: sT/sL have exponent 0.2 each, so +10% sT = +2% S.

