# Provenance of the final weights: `suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8`

| Field | Value | Evidence |
|---|---|---|
| HF repo / revision | `suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8` @ `1909e853b46b2618457ba4205e48a61df6a0d888` | HF API (`model_info`) |
| Created / last modified | 2026-10-24 16:14:47 / 16:16:57 UTC (during the event) | HF API |
| Visibility | **private** at time of writing; TODO (team): make public or share with organizers | HF API |
| Base model | `CMKL/MANGO1.5-Qwen3.5-9B` (official organizer weights, Apache-2.0) | `MODEL_CARD.md`, `quantization_recipe_args.json` |
| Method | llm-compressor `GPTQModifier`, scheme `W8A8` (INT8 per-channel symmetric weights, `actorder: static`, block 128, dampening 0.01; INT8 dynamic per-token symmetric activations) | `artifact_W8A8/recipe.yaml`, `artifact_W8A8/quantization_config.json` |
| Kept BF16 | `lm_head`, `embed_tokens`, `visual.*`, `linear_attn.*` (Gated-DeltaNet projections), `mtp.*` | recipe `ignore` list |
| MTP head | BF16, shipped as `model_mtp.safetensors` (465 MB); used as the speculative draft (`method: mtp`) | HF file list |
| Calibration | 512 samples, max 2048 tokens, seed 42, no custom JSONL (`calib_jsonl: null`), built-in mix: WangchanThaiInstruct 45% (CC-BY-SA-4.0), ultrachat_200k 35% (MIT), Open-Platypus 20% (license: TODO, none declared in Hub metadata). No test questions. | `artifact_W8A8/quantization_recipe_args.json`, `README.md` (Calibration data) |
| Format / size | compressed-tensors `int-quantized` (version 0.19.0), `model.safetensors` 13.5 GB | `config.json`, HF file list |
| Libraries recorded in the artifact | compressed-tensors 0.19.0, transformers 5.14.1 | artifact `config.json` |
| Serving kernel on A100 | vLLM 0.32.0 `CutlassInt8ScaledMMLinearKernel` for `CompressedTensorsW8A8Int8` | `experiments/E045/server.log` |

## Reproduce

```bash
cd quantization
bash setup_env.sh && source ~/mango-quant/env.sh        # quant env (see README.md here; separate from the serving venv)
python quantize_mango.py --scheme w8a8                   # = defaults recorded in artifact_W8A8/quantization_recipe_args.json
hf upload <you>/MANGO1.5-Qwen3.5-9B-W8A8 out/MANGO1.5-Qwen3.5-9B-W8A8 --private
```

The files in this folder (`README.md`, `quantize_mango.py`, `setup_env.sh`, `serve.sh`, `eval_ab.sh`, `compare.py`) are vendored
from https://github.com/suppakorn-tae/mango-quant at commit `6ffc9c959e83b5461dd9ef82987b84d2a99e466c`
(2026-10-24 17:08 UTC, author Tae). `artifact_W8A8/` holds the recipe, args, quantization config and model card
copied from the uploaded artifact itself.

**Caveats (TODO for the team):**
- **The vendored toolkit may not be the exact version that made the artifact.** The commit is 54 minutes later than the artifact upload. The artifact records transformers 5.14.1, while this `setup_env.sh` pins `transformers>=5.15,<=5.17`. The artifact's own `recipe.yaml` and `quantization_recipe_args.json` are the authoritative record of settings. Please record the exact llmcompressor / torch / transformers versions and the command that produced rev `1909e853`.
- **The vendored `serve.sh` / `eval_ab.sh` are the toolkit's own A/B helpers.** The submission is served with `scripts/serve_final.sh` instead. The toolkit's A/B Thai-benchmark results, if any, are not in this repo (TODO).
