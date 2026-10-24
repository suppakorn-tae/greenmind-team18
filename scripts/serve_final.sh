#!/usr/bin/env bash
# FINAL submission server (= configs/E046.sh = configs/champion.sh, OFFICIAL Final S 99.38, attempt 20261024T164844-c109c6).
#
# Engine : vLLM 0.32.0 (+ serving/vllm_think_budget.py: server-side default thinking budget)
# Weights: suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8 rev 1909e853 (team-made INT8 W8A8 GPTQ derivative of
#          CMKL/MANGO1.5-Qwen3.5-9B; MTP head kept BF16 and used as the speculative draft)
# Power  : GPU default power limit (400 W on A100-SXM4-40GB), no clock lock -> run `scripts/power.sh reset` first.
# API    : OpenAI-compatible /v1/chat/completions on :8000, streaming + usage, tool_choice=auto (qwen3_coder parser),
#          reasoning split (qwen3 parser). Served model name: "mango".
#
# Overridable env (defaults reproduce our VM exactly):
#   VLLM_VENV     python venv with vllm==0.32.0              [~/vllm-env]
#   MODEL_PATH    local snapshot dir or HF repo id            [local HF-cache snapshot of the W8A8 repo, else repo id]
#   THINK_BUDGET  default thinking_token_budget per request   [3072]
#   PORT          [8000]
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
VLLM_VENV="${VLLM_VENV:-$HOME/vllm-env}"
HF_REPO=suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8
HF_REV=1909e853b46b2618457ba4205e48a61df6a0d888
PORT="${PORT:-8000}"

# FlashInfer's sampler JITs at first use and needs ninja + nvcc on PATH (E001 crash without it).
export PATH="$VLLM_VENV/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

if [ -z "${MODEL_PATH:-}" ]; then
  SNAP="${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${HF_REPO/\//--}/snapshots/$HF_REV"
  if [ -d "$SNAP" ]; then
    MODEL_PATH="$SNAP"
    export HF_HUB_OFFLINE=1   # serve the local snapshot; no network / token needed at runtime
  else
    MODEL_PATH="$HF_REPO"     # downloads on first start (needs HF_TOKEN while the repo is private)
  fi
fi

export THINK_BUDGET="${THINK_BUDGET:-3072}"
exec "$VLLM_VENV/bin/python" "$REPO_DIR/serving/vllm_think_budget.py" serve "$MODEL_PATH" \
  --served-model-name mango --host 0.0.0.0 --port "$PORT" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --gpu-memory-utilization 0.92 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 \
  --enable-chunked-prefill --no-enable-prefix-caching --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"'"$MODEL_PATH"'","method":"mtp","num_speculative_tokens":2}'
