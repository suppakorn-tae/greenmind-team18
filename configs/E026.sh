#!/usr/bin/env bash
# E026: BF16 + MTP(k=2) + E001 tuned flags (no prefix caching, SSM cache bf16, language-model-only, len 41984). Power: 400 W (default).
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
M=CMKL/MANGO1.5-Qwen3.5-9B
exec ~/vllm-env/bin/vllm serve $M --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --long-prefill-token-threshold 768 --enable-chunked-prefill --no-enable-prefix-caching --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"'$M'","method":"mtp","num_speculative_tokens":2}'
