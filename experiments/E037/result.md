# E037 result — rejected
Online FP8 (vLLM: fp8_per_tensor, MarlinFP8ScaledMMLinearKernel weight-only on Ampere).
val 20261024-205259-e037: 47/100, eta 0.97x, p95 TPOT 65 ms, fails ttft 46/tpot 16, acc 80% (agent 3/20).
final 20261024-205419-e037: S 0.0, 61/300, p95 TPOT 131 ms, p95 TTFT 10.4 s, 143 running/33 waiting; acc 78.7% (reason 0.89, long 0.47, agent 0.23).
Marlin dequant kernels are far slower than BF16 tensor-core GEMMs at serving batch sizes on A100; per-tensor FP8 also costs accuracy. Quantization (W4 or W8) is a dead end on this GPU.
