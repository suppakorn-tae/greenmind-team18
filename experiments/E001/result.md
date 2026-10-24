# E001 result — rejected (AWQ-W4A16 + MTP k=3, kit "awq-mtp" profile)
- Launch 1 crashed: FlashInfer sampler JIT needs `ninja`/`nvcc` on PATH -> all configs export PATH/CUDA_HOME since.
- val cold (fresh server) `20261024-134958-e001`: 45/100 per arrival, fails ttft 51 / tpot 12 / cutoff 3, p95 TTFT 3.19 s, p95 TPOT 63 ms, 272.5 J/req, accuracy 77% (agent 0.10).
- val warm repeat `20261024-135206-e001-2`: 132/138 turns, p95 TPOT 20 ms, 227 J/req, eta 1.85x — later shown to be a prefix-cache replay artifact (E002).
- Lesson: the organizer-provided AWQ checkpoint (CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp) loses agentic accuracy (2/20 deterministic, see E004) and Marlin W4A16 is slow at serving batch sizes on A100.
