#!/usr/bin/env bash
# champion (= E046 = E045 served as "mango"; weights: team W8A8 INT8 checkpoint suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8 as target + MTP draft, thinking budget user-authorized 2026-10-24): BF16 + MTP(k=2) + E001 tuned flags (no prefix caching, SSM cache bf16, language-model-only, len 41984). Power: 400 W (default).
export PATH=$HOME/vllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
export HF_HUB_OFFLINE=1   # private repo: serve the local snapshot, no token at runtime
M=$HOME/.cache/huggingface/hub/models--suppakorn-k--MANGO1.5-Qwen3.5-9B-W8A8/snapshots/1909e853b46b2618457ba4205e48a61df6a0d888
THINK_BUDGET=${THINK_BUDGET:-3072} exec ~/vllm-env/bin/python "$(dirname "$(readlink -f "$0")")/../serving/vllm_think_budget.py" serve $M --served-model-name mango --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 --enable-chunked-prefill --no-enable-prefix-caching --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"'$M'","method":"mtp","num_speculative_tokens":2}'
