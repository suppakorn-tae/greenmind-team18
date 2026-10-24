# E045 result — CHAMPION (local), official pending
E030 config with the team's W8A8 INT8 checkpoint (suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8 @ 1909e853, served from local snapshot with HF_HUB_OFFLINE=1; served name suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8). vLLM picked CutlassInt8ScaledMMLinearKernel (native INT8 tensor cores on A100). Model 12.24 GiB; KV pool 502,724 tokens (E030 475,367). MTP acceptance ~2.5-2.6.
| run | track | on-time (turns) | per-arrival | eta | J/req | p95 TTFT | p95 TPOT | acc | local S |
|---|---|---|---|---|---|---|---|---|---|
| 20261024-232535-e045 | val | 137/137 | 100/100 | 2.13x | 205 | 1.11 s | 36 ms | 81% (agent 3/20 -> A 0.05) | 4.0 |
| 20261024-232643-e045 | final | 374/433 | 245/300 | 3.07x | 175 | 1.62 s | 51 ms | 83.0% | **93.7** |
| 20261024-232930-e045-2 | val | 146/146 | 100/100 | 2.19x | 213 | 1.09 s | 36 ms | 78% (agent 4/20, rag 17/20) | 60.3 |
| 20261024-233040-e045-2 | final | 376/420 | 261/300 | 2.99x | 180 | 1.61 s | 48 ms | 84.0% | **95.0** |
vs E030: val per-arrival 81-83 -> 100; final local S 25.5-28.8 -> 93.7-95.0; final peak 0 waiting, KV 66% (E030: 8-17 waiting, 97-99%).
Final class accuracy (2 runs): agent 0.37/0.38 (E030 0.27-0.28), chat 1.0, long 0.93/0.93 (0.53-0.60), rag 0.93/0.93 (0.97), reason 0.89/0.91 (0.93-0.96), thai 0.90/0.93 (0.90). Overall 83-84% vs 81%.
Why: the peak bottleneck was compute-bound prefill of long prompts; INT8 W8A8 GEMMs run ~2x BF16 on A100 tensor cores (unlike AWQ/FP8 weight-only Marlin paths), so prefill-heavy steps shrink and decoders keep TPOT < 50 ms.
Watch: rag (val 17/20 in run 2) and reason slightly below BF16 — check official class retention.
