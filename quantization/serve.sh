#!/usr/bin/env bash
# Serve a MANGO checkpoint with vLLM on the RUN pod (A100-40GB, driver 580 / CUDA 13).
# First time:  INSTALL=1 bash serve.sh <model_path_or_hf_id>
# Then:        bash serve.sh <model_path_or_hf_id>          (MTP=0 to disable spec decoding)
set -euo pipefail

MODEL="${1:?model path or HF id}"
WORK="${WORK:-$HOME/mango-serve}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-32768}"        # raise only if your workload needs it
GPU_UTIL="${GPU_UTIL:-0.92}"
MTP="${MTP:-1}"                    # MTP self-speculative decoding (head kept in BF16)
SPEC_TOKENS="${SPEC_TOKENS:-3}"

export PATH="$HOME/.local/bin:$PATH"
export HF_HOME="$WORK/hf"
mkdir -p "$WORK"

if [[ "${INSTALL:-0}" == 1 ]]; then
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  uv venv -p 3.12 "$WORK/venv"
  # driver 580 runs CUDA 13 wheels, so the default vLLM wheel is fine here
  uv pip install --python "$WORK/venv/bin/python" -U vllm "lm-eval[api]"
fi
# shellcheck disable=SC1091
source "$WORK/venv/bin/activate"

ARGS=(
  --served-model-name mango
  --port "$PORT"
  --language-model-only            # skip the unused vision tower -> more KV cache
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser qwen3_coder
)
if [[ "$MTP" == 1 ]]; then
  # if your vLLM build rejects "mtp", use "qwen3_next_mtp"
  ARGS+=(--speculative-config "{\"method\": \"mtp\", \"num_speculative_tokens\": $SPEC_TOKENS}")
fi

exec vllm serve "$MODEL" "${ARGS[@]}"
