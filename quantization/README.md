# mango-quant

Quantize [`CMKL/MANGO1.5-Qwen3.5-9B`](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B), a Thai/English
fine-tune of Qwen3.5-9B, into a vLLM-ready [compressed-tensors](https://github.com/neuralmagic/compressed-tensors)
checkpoint with [llm-compressor](https://github.com/vllm-project/llm-compressor). Then serve it and A/B it
against the BF16 original before you ship.

```
setup_env.sh ──► quantize_mango.py ──► hf upload ──► serve.sh ──► eval_ab.sh ──► compare.py
   (quant pod)                                        (run pod)
```

| File | What it does |
|---|---|
| `setup_env.sh` | No-root Python 3.12 + CUDA 12.8 torch + llm-compressor env under `~/mango-quant`. Writes `env.sh`. |
| `quantize_mango.py` | Calibrates, quantizes and saves the checkpoint (FP8 / W8A8 / W4A16). Keeps the MTP head so speculative decoding still works. |
| `serve.sh` | Starts an OpenAI-compatible vLLM server, with MTP speculative decoding on by default. |
| `eval_ab.sh` | Thai accuracy (Belebele, XCOPA, XNLI, MGSM) + throughput/latency against a running server. |
| `compare.py` | Side-by-side table of several `eval_ab.sh` runs, with the delta vs. the first column. |

## Pick a scheme

| `--scheme` | Weights / activations | Size | Calibration | Time* | When to use |
|---|---|---|---|---|---|
| `w8a8` (default) | INT8 / INT8 dynamic, GPTQ | ~14 GB | yes | 30–60 min | **Main candidate on A100.** Runs real INT8 tensor-core GEMMs. |
| `fp8` | FP8 / FP8 dynamic | ~14 GB | no | ~10 min | Data-free safety net. A100 has no FP8 cores, so vLLM runs it as W8A16 (Marlin). |
| `w4a16` | INT4 (group 128, MSE observer) / BF16, GPTQ | ~8 GB | yes | 30–60 min | Only if you must fit smaller hardware. Expect measurable accuracy loss. |

\*On one A100-40GB with the default 512 calibration samples.

These parts stay in **BF16 on purpose**, whatever the scheme:

- `lm_head` and `embed_tokens`: 248k-token vocab, cheap to keep, costly to get wrong.
- `linear_attn`: Qwen3.5's Gated-DeltaNet projections are its most quantization-sensitive layers.
- `visual`: the vision tower is unused for text serving (`serve.sh` passes `--language-model-only`).
- `mtp`: the multi-token-prediction head, saved as `model_mtp.safetensors` for self-speculative decoding.

## Quick start

### 1. Quant pod: A100-40GB, 32 GB RAM, driver 550, no root

```bash
git clone https://github.com/suppakorn-tae/mango-quant ~/mango-quant
cd ~/mango-quant
bash setup_env.sh          # needs ~45 GB free: 19 GB source + 14 GB output + venv
source env.sh              # every new shell

python quantize_mango.py --scheme fp8     # optional, ~10 min, no data needed
python quantize_mango.py --scheme w8a8    # main candidate
```

Outputs land in `out/MANGO1.5-Qwen3.5-9B-{FP8,W8A8,W4A16}/`. After quantizing, the script runs a Thai and an
English test generation so you can catch a broken model before uploading. Pass `--no-sample` to skip this.

> `source env.sh python quantize_mango.py` does **not** run Python. `source` only loads the environment, so run
> the two commands separately.

### 2. Move the checkpoint

```bash
hf auth login     # or: export HF_TOKEN=...   (write-scoped token)
hf upload <you>/MANGO1.5-Qwen3.5-9B-W8A8 out/MANGO1.5-Qwen3.5-9B-W8A8 --private
```

You can also copy the folder over a shared PVC or with `kubectl cp`. Before a public upload, edit `README.md` inside the
output folder. It is the base model's card, so set `base_model: CMKL/MANGO1.5-Qwen3.5-9B` and
`base_model_relation: quantized` there.

### 3. Run pod: A100-40GB, 48 GB RAM, driver 580 / CUDA 13

```bash
INSTALL=1 bash serve.sh <you>/MANGO1.5-Qwen3.5-9B-W8A8     # first time: creates ~/mango-serve/venv
bash serve.sh <you>/MANGO1.5-Qwen3.5-9B-W8A8               # afterwards
```

The server is served as model name `mango` on `http://localhost:8000/v1`, with the `qwen3` reasoning parser and
`qwen3_coder` tool calling enabled. A private HF repo needs `HF_TOKEN` (read scope) in the environment.

| Env var | Default | Meaning |
|---|---|---|
| `PORT` | `8000` | HTTP port |
| `MAX_LEN` | `32768` | `--max-model-len`. Raising it costs KV-cache room. |
| `GPU_UTIL` | `0.92` | `--gpu-memory-utilization` |
| `MTP` | `1` | `0` turns off MTP speculative decoding |
| `SPEC_TOKENS` | `3` | Draft tokens per step when `MTP=1` |
| `WORK` | `~/mango-serve` | venv + HF cache location |

### 4. A/B against BF16

Serve each checkpoint in turn and evaluate it under its own label:

```bash
bash serve.sh CMKL/MANGO1.5-Qwen3.5-9B &       bash eval_ab.sh bf16 CMKL/MANGO1.5-Qwen3.5-9B
# stop the server, then
bash serve.sh <you>/MANGO1.5-Qwen3.5-9B-W8A8 & bash eval_ab.sh w8a8 <you>/MANGO1.5-Qwen3.5-9B-W8A8

python compare.py results/bf16 results/w8a8 results/fp8
```

`eval_ab.sh` waits for the server, then runs:

- **Log-likelihood tasks:** `belebele_tha_Thai`, `xcopa_th`, `xnli_th`. These mirror MANGO's ThaiLLM eval and are very
  sensitive to quantization noise, so they catch damage cheaply.
- **Generation:** `mgsm_native_cot_th`, greedy, up to 4096 new tokens.
- **Speed:** `vllm bench serve` with 1024-in / 512-out random prompts at `CONC=32`.

`LIMIT=200` gives a quick partial pass. `OUT=` changes the results root.

**Ship rule of thumb:** pick the smallest checkpoint whose Thai scores stay within ~1 point of BF16, on these
tasks *and* on your own grader.

## Calibration data

For `w8a8` and `w4a16`, the script builds 512 samples (`--num-samples`) by default, each rendered with MANGO's own chat
template and system prompt:

| Source | Share | Why |
|---|---|---|
| [`airesearch/WangchanThaiInstruct`](https://huggingface.co/datasets/airesearch/WangchanThaiInstruct) | 45% | Thai instructions |
| [`HuggingFaceH4/ultrachat_200k`](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 35% | English multi-turn chat |
| [`garage-bAInd/Open-Platypus`](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) | 20% | STEM / reasoning |

Samples shorter than 32 tokens are dropped, and samples longer than `--max-seq-len` (2048) are truncated.

**For the best accuracy on your task, add your real traffic.** One JSON object per line, in any of these shapes:

```json
{"prompt": "สรุปบทความนี้ให้หน่อย ...", "response": "..."}
{"prompt": "..."}
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"text": "raw text, used as-is without the chat template"}
```

```bash
python quantize_mango.py --scheme w8a8 --calib-jsonl /path/to/your_prompts.jsonl --custom-frac 0.5
```

`--custom-frac` sets how much of `--num-samples` comes from your file (default half). The built-in mix fills the rest.
A few hundred representative examples is plenty. A missing file fails immediately, before any weights load.
`*.jsonl` is git-ignored so private prompts don't get committed by accident.

## `quantize_mango.py` options

| Flag | Default | Notes |
|---|---|---|
| `--model` | `CMKL/MANGO1.5-Qwen3.5-9B` | HF id or local path |
| `--scheme` | `w8a8` | `w8a8` \| `fp8` \| `w4a16` |
| `--out` | `$WORK/out/<model>-<SCHEME>` | Output directory |
| `--num-samples` | `512` | Calibration samples. Lower it if RAM is tight. |
| `--max-seq-len` | `2048` | Calibration sequence length |
| `--calib-jsonl` | none | Your own prompts (see above) |
| `--custom-frac` | `0.5` | Share of samples taken from `--calib-jsonl` |
| `--seed` | `42` | Dataset shuffling |
| `--act-offload` | `cuda:0` | Where per-layer activations live during calibration. `cpu` saves VRAM but costs RAM. |
| `--no-sample` | off | Skip the post-quant test generation |
| `--patch-aux DIR` | none | Only add missing processor/chat-template files and re-check MTP in an existing output |

Each output folder also records `recipe.yaml` (the llm-compressor recipe) and `quantization_recipe_args.json` (the
exact CLI args and ignore list), so you can reproduce it.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `torch.cuda.is_available() == False` on the quant pod | Reinstall torch from `https://download.pytorch.org/whl/cu126` |
| Process OOM-killed (host RAM) during calibration | `--num-samples 256`. Activations are already on GPU with the default `--act-offload cuda:0`. |
| CUDA OOM during calibration | `--act-offload cpu --num-samples 256` |
| `--calib-jsonl file not found` | The path is yours to supply. Drop the flag to use the built-in mix. |
| vLLM rejects `"method": "mtp"` | Edit `serve.sh` to use `qwen3_next_mtp` |
| MTP lowers throughput at high concurrency | Draft work is wasted when the GPU is saturated. Benchmark `MTP=0` vs `MTP=1` at your real `CONC`. |
| `[mtp] 0 MTP tensors ... WARNING` | Run `python quantize_mango.py --patch-aux out/<folder>` |
| Older output missing `preprocessor_config.json` / chat template | Same `--patch-aux` fix |
| `HF_HUB_ENABLE_HF_TRANSFER is deprecated` warning | Harmless. Replace it with `export HF_XET_HIGH_PERFORMANCE=1` in `env.sh` if it bothers you. |

## Requirements

- **Quant:** NVIDIA GPU with ≥40 GB VRAM, ≥32 GB RAM, ~45 GB disk, CUDA 12.8-capable driver. Uses torch 2.10–2.14,
  `llmcompressor>=0.14`, `compressed-tensors>=0.19` and `transformers` 5.15–5.17.
- **Serve:** recent vLLM with Qwen3.5 + compressed-tensors support. `INSTALL=1` installs the latest vLLM and `lm-eval[api]`.

The base model and its quantized derivatives are Apache-2.0, following
[Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE).
