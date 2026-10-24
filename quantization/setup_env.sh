#!/usr/bin/env bash
# No-root environment for quantizing MANGO1.5 on the QUANT pod
# (A100-40GB, driver 550, 32 GB RAM). Everything lives under $WORK.
# Usage:  bash setup_env.sh        then:  source $WORK/env.sh
set -euo pipefail

WORK="${WORK:-$HOME/mango-quant}"
mkdir -p "$WORK"/{hf,out,venv}

# uv: single static binary into ~/.local/bin, no root needed
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Python 3.12 managed by uv (container python may be too old)
uv venv -p 3.12 "$WORK/venv/q"
# shellcheck disable=SC1091
source "$WORK/venv/q/bin/activate"

# torch FIRST from the cu128 index: this pod's driver (550 / CUDA 12.8 compat)
# cannot run CUDA-13 wheels. llmcompressor 0.14 needs torch 2.10-2.14.
uv pip install "torch>=2.10,<=2.14" --index-url https://download.pytorch.org/whl/cu128
uv pip install "llmcompressor>=0.14.0" "compressed-tensors>=0.19.0" \
  "transformers>=5.15,<=5.17" datasets accelerate huggingface_hub hf_transfer

cat > "$WORK/env.sh" <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
source "$WORK/venv/q/bin/activate"
export HF_HOME="$WORK/hf"            # keep 19 GB of weights off the small root fs
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EOF

source "$WORK/env.sh"
python - <<'PY'
import torch, transformers, llmcompressor, compressed_tensors
print("torch", torch.__version__, "cuda", torch.version.cuda, "available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA not visible -> try the cu126 index for torch"
print(torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/2**30,1), "GiB")
print("transformers", transformers.__version__, "| llmcompressor", llmcompressor.__version__,
      "| compressed-tensors", compressed_tensors.__version__)
PY
df -h "$WORK" | tail -1
echo "Need ~45 GB free in $WORK (19 GB source + 14 GB output + venv)."
echo "OK -> source $WORK/env.sh"
