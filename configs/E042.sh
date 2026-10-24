#!/usr/bin/env bash
# E042: TensorRT-LLM 1.4.0rc1 (PyTorch backend, venv ~/trtllm-env), official BF16 weights, MTP 2 draft tokens,
# mamba ssm bf16, chunked prefill 4096, no block reuse, no thinking budget yet (bring-up). Power: 400 W (default).
export PATH=$HOME/trtllm-env/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
exec ~/trtllm-env/bin/trtllm-serve serve CMKL/MANGO1.5-Qwen3.5-9B --host 0.0.0.0 --port 8000 \
  --reasoning_parser qwen3_5 --tool_parser qwen3_coder \
  --config ~/repo/configs/trtllm/E042.yaml
