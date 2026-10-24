#!/usr/bin/env bash
# E041: SGLang 0.5.22 (separate venv ~/sglang-env), official BF16 weights, NEXTN/MTP 2 draft steps, mamba ssm bf16,
# chunked prefill 4096, no radix cache, no thinking budget yet (engine bring-up). Power: 400 W (default).
export PATH=$HOME/sglang-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
M=CMKL/MANGO1.5-Qwen3.5-9B
exec ~/sglang-env/bin/python -m sglang.launch_server --model-path $M --host 0.0.0.0 --port 8000 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --language-only \
  --context-length 41984 --mem-fraction-static 0.85 --chunked-prefill-size 4096 \
  --mamba-ssm-dtype bfloat16 --disable-radix-cache \
  --speculative-algorithm NEXTN --speculative-num-steps 2 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3
