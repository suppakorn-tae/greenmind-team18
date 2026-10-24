#!/usr/bin/env bash
# E001: AWQ-W4A16 + MTP(k=3), prefix caching (mamba align), language-model-only, len 41984. Power: 400 W (default).
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
AWQ=CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp
exec ~/vllm-env/bin/vllm serve $AWQ --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
  --speculative-config '{"model":"'$AWQ'","method":"mtp","num_speculative_tokens":3}' \
  --mamba-cache-mode align
