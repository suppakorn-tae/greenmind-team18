#!/usr/bin/env bash
# E039 (= E038 + transition phrase on forced reasoning end; thinking budget user-authorized 2026-10-24): BF16 + MTP(k=2), no prefix caching, SSM cache bf16, language-model-only, len 41984. Power: 400 W (default).
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
M=CMKL/MANGO1.5-Qwen3.5-9B
export THINK_BUDGET=${THINK_BUDGET:-1536}
RCFG='{"reasoning_end_str": "\n\nConsidering the limited time, I have to stop thinking and give the final answer directly now.\n</think>\n\n"}'
exec ~/vllm-env/bin/python ~/repo/tools/gray/vllm_think_budget.py serve $M --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --reasoning-config "$RCFG" \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 --enable-chunked-prefill --no-enable-prefix-caching --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"'$M'","method":"mtp","num_speculative_tokens":2}'
