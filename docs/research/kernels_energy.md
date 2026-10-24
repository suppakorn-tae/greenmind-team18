# GPU/kernel and engine knobs for decode/prefill speed and energy: MANGO1.5-Qwen3.5-9B (GDN hybrid) on 1x A100-SXM4-40GB, vLLM 0.32.0

Source conventions:
- **Local vLLM 0.32.0 source** means the installed package at `~/vllm-env/lib/python3.12/site-packages/vllm/` (commit g5df63a720), with the file path given. It was read directly, so it is authoritative for this exact build.
- **Local logs** means server logs under `~/repo/experiments/`.
- Web sources are linked.

**Framing point (inference; it explains most of the earlier results).** At peak, a decode step with about 130-140 sequences and MTP k=2 runs every weight GEMM with M ≈ 390-420 verify tokens. Most peak steps also carry up to 4096 chunked-prefill tokens.
- A weight GEMM's arithmetic intensity is about M FLOP/byte. A100's BF16 ridge point is about 312 TFLOPS / 1.55-2.0 TB/s ≈ 156-200 FLOP/byte.
- So at peak, the linear layers are **compute-bound, not memory-bound**. Mixed prefill+decode steps are even more so.
- Rough check using your numbers:
  - Prefill at 5-10K tok/s on about 7B non-embedding params is about 2 × 7e9 × 7.5K ≈ 105 TFLOP/s.
  - Decode at about 140 seqs / 75 ms TPOT, with about 3/2.61 verify tokens per accepted token, adds about 30 TFLOP/s.
  - Together that is about 135 TFLOP/s, roughly 43% of the BF16 peak. That is a realistic ceiling for a hybrid model with GDN/attention overheads.
- This one fact explains three observations:
  - (a) The 1215 MHz clock lock and the power caps cost SLO about 1:1 with energy.
  - (b) Marlin W4A16 and W8A16 lost. They dequantize to BF16 and still run at the BF16 MMA rate, plus dequant overhead.
  - (c) The literature's "decode is frequency-insensitive" results don't transfer to this regime.
- The only lever that raises the compute ceiling is **INT8 tensor cores (W8A8)**. A100 dense INT8 is 624 TOPS, about 2x BF16.

## Gated-DeltaNet / FLA kernel performance in vLLM on A100

### Takeaway
On A100 with your flags, vLLM 0.32 already uses the fastest GDN paths available:
- the **fused CUDA GDN MTP decode kernel**;
- **Model Runner V2**;
- **async scheduling**.

GDN prefill is **Triton/FLA only**. The FlashInfer and CuteDSL GDN prefill kernels are gated to SM90/SM100/SM120. No env var or backend switch is left with meaningful upside on Ampere. Recent upstream GDN speedups are either Blackwell-only or unmerged and measured only at batch 1.

### Cited Findings
- **Prefill backend selection.**
  - `_resolve_gdn_prefill_backend` picks FlashInfer GDN prefill only on Hopper (SM90), or on Blackwell SM10.x/SM12.x with head_k_dim=128 and CUDA runtime ≥13.
  - CuteDSL is opt-in and SM10.x only. Everything else, including A100 (SM80), returns `"triton"` (FLA).
  - The request goes through `additional_config["gdn_prefill_backend"]`, default `"auto"`. The log hint suggests `--gdn-prefill-backend triton` to skip the JIT on Hopper.
  - Source: local vLLM 0.32.0 source, `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` (lines ~93-192).
- **Your logs confirm the kernel choices:**
  - "Using Triton/FLA GDN prefill kernel (requested=auto, head_k_dim=128)" (57 runs).
  - "GDN spec decode kernel: cuda" (56 runs).
  - "Using V2 Model Runner".
  - "Async scheduling is already on (mtp is in EagleModelTypes)".
  - "Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION']".
  - "Using FlashInfer for top-p & top-k sampling."
  - Source: local logs (`~/repo/experiments/E0xx/*`).
- **Decode env vars that exist in 0.32:**
  - `VLLM_GDN_DECODE_KERNEL` = `cuda` (default) or `triton`. "cuda uses the fused decode kernel where supported and falls back to triton otherwise."
  - `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE` (default 1).
  - The fused CUDA MTP decode kernel requires all of: a BF16 model, K=V=128, SiLU/sigmoid gating, non-interleaved GQA layout (true for Qwen3.5, false for Qwen3-Next), **BF16 conv cache**, **BF16 or FP32 recurrent state**, compute capability ≥8.0, and `MAX_FUSED_GDN_MTP_TOKENS = 8`.
  - So your `--mamba-ssm-cache-dtype bfloat16` keeps you on the fused path. A hypothetical FP16 or FP8 SSM state would drop you to Triton.
  - Source: local vLLM 0.32.0 source, `envs.py` (lines ~1230-1241) and `qwen_gdn_linear_attn.py` (lines ~89-90 and ~594-646).
- **Model dims matter for kernel eligibility.** MANGO config: linear_key/value_head_dim 128/128, 16 K-heads / 32 V-heads, attention 16 Q / 4 KV heads with head_dim 256, hidden 4096, intermediate 12288, vocab 248320, 24 linear_attention + 8 full_attention layers (`full_attention_interval` 4). Source: local model config.json (HF cache).
- **Unsloth A100 measurements (vLLM 0.31, Qwen3.5-2B, A100 40GB).**
  - An opt-in Triton ReplaySSM GDN decode ran at 0.804x (batch 8) and 0.682x (batch 64) of the default V2 path. So the default V2 path is faster.
  - "Switching to Model Runner V1 alone costs 23% at 64 sequences on A100" (V1 without ReplaySSM: 0.766x).
  - The isolated GDN ring kernel is slower than the packed decode kernel below about 100 sequences: 0.67x at 32 seqs, 1.63x at 128 seqs (ratio direction not stated explicitly).
  - Source: [unsloth-zoo PR #1761](https://github.com/unslothai/unsloth-zoo/pull/1761).
- **Upstream ReplaySSM.** Upstream GDN ReplaySSM work (vLLM PR #48792) was closed unmerged. Source: [unsloth-zoo PR #1761](https://github.com/unslothai/unsloth-zoo/pull/1761).
- **vLLM PR #62961 "[Perf] Speed up Qwen GDN decode and MRV2 attention metadata".**
  - What it does: folds the `in_proj_ba` GEMM into `in_proj_qkvz` for unquantized projections, reads the b/a gates as strided views (removes a copy kernel per layer), runs the gated RMSNorm in place for spec decode, and shares MRV2 attention metadata across KV groups.
  - Measured: +5.8% without speculation and +5.6% with MTP-3, but only at **batch 1 on 4x B300 TP4**.
  - Status: open, not merged. A reviewer raised a reload-safety concern.
  - Source: [vLLM PR #62961](https://github.com/vllm-project/vllm/pull/62961).
- **vLLM's Qwen3.5 performance blog.**
  - The GDN prefill gains are Blackwell-only: FlashInfer PR #3001, up to about 5.78-5.92x kernel and 1.13x end-to-end prefill, enabled by vLLM PR #40717.
  - Generic flags it credits: `--async-scheduling`, `--mamba-ssm-cache-dtype bfloat16`, and `--language-model-only`, which "unlocks the fused QK-norm + RoPE + gate path in attention". You already use all three.
  - Larger `--max-num-batched-tokens` (2x ISL; about +8% TPS at very high concurrency).
  - `--max-cudagraph-capture-size` raised to 640-768 only at concurrency ≥4096, with the note "not certain this is actually required".
  - Source: [vLLM blog, Qwen3.5 25K TPS](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps).

### Inferences
- There is no GDN-side env/backend switch to flip on A100.
  - Setting `VLLM_GDN_DECODE_KERNEL=triton` would be a regression.
  - The FlashInfer GDN prefill can't be enabled on SM80 without patching the gate. Even if forced, FlashInfer's GDN prefill kernel targets Hopper/Blackwell instructions (TMA/wgmma), so it likely won't compile or run on SM80. That is an inference, not tested.
- The only GDN-specific code-level idea with plausible A100 upside is a backport of #62961's in_proj_ba→qkvz fold and gate-copy removal.
  - At batch 130+ the fold saves one small GEMM launch plus a copy per GDN layer (24 layers).
  - My estimate is ≤1-3% of step time at peak, because steps are compute-dominated. That is low value for the risk.
- Per-sequence GDN state traffic is non-trivial at peak (estimate):
  - Each sequence holds 32 V-heads x 128 x 128 x 2 B = 1 MiB per layer, x 24 layers = 24 MiB.
  - At 140 seqs that is about 3.4 GB read plus about 3.4 GB written per step, about 4-5 ms at about 1.5 TB/s.
  - BF16 state (already set) halves this versus FP32. No further reduction is available without accuracy risk.

### Gaps
- I found no A100-specific benchmark of FLA Triton chunk-prefill (chunk size 64) versus alternatives, and no Triton autotune config overrides for SM80 in vLLM 0.32's vendored FLA (`vllm/third_party/flash_linear_attention/ops/`).
- I found no measurement of the fused CUDA GDN MTP decode kernel versus Triton on A100 at batch 64-256.

## Attention backend, CUDA graphs, torch.compile, sampler

### Takeaway
These knobs have essentially **no remaining upside** at your peak, for three reasons:
- Peak steps are large mixed prefill+decode batches (up to 4096+~420 tokens). Those exceed the 512-token CUDA-graph capture limit and are compute-dominated, so launch overhead is negligible.
- Only 8 of 32 layers are attention.
- `-O3` is defined identically to the default `-O2` in 0.32.

### Cited Findings
- **Optimization levels.**
  - In 0.32, `OPTIMIZATION_LEVEL_02` and `OPTIMIZATION_LEVEL_03` are identical dicts: same pass_config fusions, `cudagraph_mode: FULL_AND_PIECEWISE`, `use_inductor_graph_partition: False`, FlashInfer autotune on. So `-O3` gives no extra optimization over the default.
  - `fuse_attn_quant` is enabled only when the model is quantized (`IS_QUANTIZED`).
  - `fuse_norm_quant` and `fuse_act_quant` are conditional (`enable_norm_fusion` / `enable_act_fusion`). They matter only for quantized (e.g., INT8/FP8) models.
  - Source: local vLLM 0.32.0 source, `config/vllm.py` (lines ~247-327).
- **Full CUDA-graph size limit with spec decode.** The largest uniform-decode batch a full CUDA graph can cover is `max_num_seqs * uniform_decode_query_len`, where the query length is `1 + num_speculative_tokens` (3 for k=2). With the default `max_cudagraph_capture_size` 512, full graphs cover pure-decode steps up to about 170 sequences. Source: local vLLM 0.32.0 source, `config/vllm.py` (lines ~660-685).
- **Spec-decode speedup at high batch, from other platforms.**
  - Huawei CloudMatrix384: MTP's relative decode-throughput gain shrinks as batch grows (6-49% across batch sizes, larger at small batch). Per-layer decode latency rose about 44% with MTP at batch 96. Source: [arXiv 2506.12708](https://arxiv.org/pdf/2506.12708).
  - SwiftSpec: verification cost grows with batch. Going from batch 8 to 16 cost +29% inference time on Llama3-70B TP4. Source: [arXiv 2506.11309](https://arxiv.org/pdf/2506.11309).
- **`--stream-interval 100`** reduces frontend overhead at very high concurrency but "buffers output in 100-token chunks". Source: [vLLM blog](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps).
- **Dynamic speculative depth (new knob in 0.32, not yet tried).**
  - `SpeculativeConfig.num_speculative_tokens_per_batch_size: list[(range_start, range_end, k)]` sets an inclusive batch-size schedule. The scheduler looks up k per step from the running batch size: `dynamic_sd_lookup[batch_size]` in `v1/core/sched/scheduler.py` (~line 1487).
  - Values are capped at `num_speculative_tokens`, the first range must start at 1, and gaps carry the previous k forward.
  - On the V1 runner it forces `cudagraph_mode=PIECEWISE`. The override is skipped when `use_v2_model_runner` is true, which is your case, so full CUDA graphs are kept.
  - It is disabled under data parallelism (irrelevant here).
  - Example: `--speculative-config '{"model":"CMKL/MANGO1.5-Qwen3.5-9B","method":"mtp","num_speculative_tokens":2,"num_speculative_tokens_per_batch_size":[[1,96,2],[97,256,1]]}'`
  - Source: local vLLM 0.32.0 source, `config/speculative.py` (line 491), `v1/spec_decode/dynamic/utils.py`, and `config/vllm.py` (`_maybe_override_dynamic_sd_cudagraph_mode`, ~1039-1056).
- **Prior dynamic-length work.** An earlier per-sequence dynamic speculation length in vLLM 0.8.4 (DSDE) had to run in eager mode because changing the length otherwise needed CUDA-graph re-capture. Source: [arXiv 2509.01083](https://arxiv.org/pdf/2509.01083).
- **`enable_adaptive_verification`** is "Currently only supported for method='dspark'", so it is not usable with MTP. Source: local vLLM 0.32.0 source, `config/speculative.py` (line ~551).

### Inferences
- **Dynamic MTP depth is the only engine knob here with a plausible peak-TPOT gain.** The estimate below uses your measured acceptance of 0.87 at position 1 and 0.74 at position 2.
  - k=2 spends 3 verify tokens per sequence per step for about 2.61 accepted tokens. k=1 spends 2 for about 1.87.
  - If peak steps are compute-bound (cost ∝ tokens), k=1's TPOT is about (2/1.87)/(3/2.61) ≈ 0.93x of k=2. That is roughly **7% lower TPOT at peak**, and it also frees about 1/3 of the decode verify compute for prefill.
  - If steps were memory-bound (fixed cost), k=2 would win by about 28%.
  - Reality sits between these, so measure it. Static k=1 lost on J/req (+23%) in your runs because most of the run is off-peak. A schedule like `[[1,96,2],[97,256,1]]` keeps k=2 off-peak.
  - Tune the threshold, e.g. 64/96/128. Verify that the MTP state slots (k+1) don't change memory admission.
- **CUDA graph tuning.**
  - Raising `max_cudagraph_capture_size` above 512 only helps pure-decode steps with more than about 170 sequences. Your peak is about 130-140, and most peak steps also carry prefill.
  - Mixed steps run piecewise/compiled. At 4K-token steps (≥100 ms), launch overhead is under 1%.
  - Lowering capture sizes saves memory (a few hundred MB), not time.
- **Attention backend.** FLASH_ATTN on A100 is FlashAttention-2, which supports head_dim 256. The alternatives (FlashInfer, Triton) have no evidence of being faster on SM80 for this shape. FlashInfer was only forced when you tried FP8 KV, and that caused the CUDA graph downgrade. Attention is 8 of 32 layers, so even a 10% attention-kernel gain would be about 1-3% of step time (estimate).
- **Sampler.** The FlashInfer top-p/top-k sampler is already active (`VLLM_USE_FLASHINFER_SAMPLER` default True).
  - The 248K vocab makes logits large: about 420 x 248320 fp32 ≈ 0.4 GB per step at peak.
  - The LM head GEMM is about 1.0B params x 420 tokens ≈ 0.86 TFLOP per step, a few percent of step compute.
  - No sampler knob changes this.
- **`--stream-interval`** buffers tokens. It would distort the TTFT/TPOT your grader measures, so it is risky and not recommended.

### Gaps
- I found no measured A100 comparison of FLASH_ATTN vs FLASHINFER vs TRITON_ATTN for hybrid GDN models.
- I found no published measurement of `num_speculative_tokens_per_batch_size` with MTP. It needs a local test.
- I didn't verify whether `fuse_norm_quant`/`fuse_act_quant` cover INT8 dynamic per-token quant on CUDA in 0.32 (the `enable_norm_fusion` body wasn't inspected).

## Weight-level options fast on Ampere (INT8 W8A8 vs Marlin W4A16/W8A16)

### Takeaway
**INT8 W8A8 (compressed-tensors, GPTQ ± SmoothQuant) is the single highest-upside remaining lever.** It is the only format that runs GEMMs on A100's INT8 tensor cores (about 2x the BF16 peak). Your peak is GEMM-compute-bound, which is exactly where the Marlin weight-only kernels lost.
- A W8A8 recipe and checkpoint already exist for base Qwen3.5-9B, with 99.4-100.2% accuracy recovery.
- That checkpoint is a third-party derivative of base Qwen3.5, not MANGO. Under your rules you must quantize the MANGO weights yourselves with the same recipe.
- I found no A100 8B-class W8A8 speed number. Expected gains are estimates: perhaps 1.15-1.35x on peak step time.

### Cited Findings
- **vLLM 0.32 INT8 support on A100.**
  - `CompressedTensorsW8A8Int8.get_min_capability()` returns 75 ("turing and up"), so it is supported on A100.
  - CUDA INT8 kernel candidates: `CutlassInt8ScaledMMLinearKernel`, `TritonInt8ScaledMMLinearKernel`, `HummingInt8ScaledMMLinearKernel` (`_POSSIBLE_INT8_KERNELS`).
  - Source: local vLLM 0.32.0 source, `compressed_tensors/schemes/compressed_tensors_w8a8_int8.py` and `model_executor/kernels/linear/__init__.py` (~line 431).
- **vLLM docs:** INT8 W8A8 computation is supported on compute capability >7.5 (Turing, Ampere, Ada, Hopper, Blackwell). Source: [vLLM INT8 W8A8 docs](https://docs.vllm.ai/en/v0.8.5/features/quantization/int8.html).
- **Existing W8A8 checkpoint for base Qwen3.5-9B** ([RedHatAI/Qwen3.5-9B-quantized.w8a8 model card](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w8a8)):
  - Scheme: INT8 weights and activations with LLM Compressor GPTQ (`targets="Linear"`, `dampening_frac=0.01`).
  - Ignore list: `re:.*lm_head`, `re:.*embed_tokens$`, `re:.*visual.*`, `re:.*model.visual.*`, `re:.*linear_attn.*`.
  - The MTP tensors are copied back from the source checkpoint with `save_mtp_tensors_to_checkpoint(source_model=..., dest_dir=...)`.
  - Size: 19.3 GB → 14.0 GB, because the vision encoder, embeddings and linear-attention layers stay BF16.
  - Accuracy, BF16 → W8A8: GSM8k-Platinum 94.4 → 94.3; MMLU-Pro 82.4 → 82.0; IFEval prompt-strict 89.5 → 89.6; Math-500 85.2 → 85.3; AIME25 85.4 → 85.4; GPQA-D 82.2 → 82.3. That is 99.4-100.2% recovery.
  - Built with vLLM 0.18.1. Serve with `vllm serve ... --reasoning-parser qwen3 --language-model-only`.
- **llm-compressor docs:** the Qwen3.5 page lists `RedHatAI/Qwen3.5-9B-quantized.w8a8` and a 4B variant. The examples need `transformers >= v5`. Source: [llm-compressor Qwen3.5 key-models page](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/).
- **Generic W8A8 INT8 recipe:** `SmoothQuantModifier(smoothing_strength=0.8)` + `GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])`. Source: [llm-compressor W8A8 INT8 example](https://docs.vllm.ai/projects/llm-compressor/en/latest/examples/quantization_w8a8_int8/).
- **"Give Me BF16 or Give Me Death" (Red Hat / Neural Magic):**
  - W8A8-INT, when properly tuned, costs 1-3% accuracy. Llama-3.1-8B-Instruct W8A8-INT: Arena-Hard 27.2 vs 25.8 BF16, HumanEval 67.1 vs 67.3, academic-average recovery 101.5%. R1-Distill-Llama-8B W8A8-INT: 99.6% reasoning recovery (AIME24 49.1 vs 49.3, MATH-500 90.0 vs 90.2).
  - Performance: "W8A8 dominates in asynchronous continuous batching", while W4A16 wins in synchronous/latency deployment.
  - 8B async on **A6000** (Ampere): INT8 2.2 QPS vs BF16 1.5 (1.38x).
  - 70B on 4x **A100**: INT8 2.4 vs BF16 1.4 QPS (1.87x overall).
  - The paper has **no A100 numbers for 8B**.
  - Source: [arXiv 2411.02355](https://arxiv.org/html/2411.02355v3); [Red Hat blog](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms).
- **Granite-3.1-8B W8A8 card:** "up to 1.6x speedup single-stream, up to 1.7x multi-stream", with vLLM 0.6.6. The hardware isn't specified per number. Source: [RedHatAI granite-3.1-8b-base W8A8 card](https://huggingface.co/RedHatAI/granite-3.1-8b-base-quantized.w8a8).
- **Third-party summary (not original):** one page claims W8A8 gives 1.3-1.7x throughput on A100 at high concurrency, but it is internally inconsistent and not primary, so treat it as unreliable. Source: [markaicode benchmark page](https://markaicode.com/benchmarks/vllm-deepseek-benchmark/).

### Inferences
- **Why INT8 should work where AWQ and FP8-Marlin failed (inference).** Marlin W4A16 and W8A16 dequantize weights to BF16 inside the kernel and use BF16 MMA. At M ≈ 400-4500 they are compute-bound with extra dequant work, so they are slower than BF16 cuBLAS, which matches your E037. CUTLASS INT8 scaled_mm uses s8 MMA at 2x the BF16 rate and also halves weight bytes.
- **FLOP coverage under the RedHat ignore list (estimate):**
  - MLP: 3 x 4096 x 12288 ≈ 151M params/layer x 32 ≈ 4.8B.
  - Full-attention projections: about 59M/layer x 8 ≈ 0.47B.
  - Both of these go INT8, about 5.3B params.
  - These stay BF16: GDN projections, about 67M/layer x 24 ≈ 1.6B (`in_proj_qkvz` 4096 → 12288, `out_proj` 4096 → 4096), plus the LM head (about 1.0B).
  - So about 75% of per-token body GEMM FLOPs (excluding the LM head) move to INT8.
  - If INT8 GEMMs achieve about 1.6-1.8x the BF16 throughput at these M, and GEMMs are about 60-75% of step time at peak, the estimated whole-step speedup is about 1.2-1.35x. This applies to both prefill chunks and decode verify.
  - TPOT p95 60-90 ms → about 47-75 ms would close much of the gap to the 50 ms SLO.
  - J/req should also fall, since there are fewer cycles per token and INT8 MMA uses less energy per op. Not measured.
- **Extra gain (inference).** Also quantizing `linear_attn.in_proj_qkvz`/`out_proj` (deviating from the RedHat ignore list) would push coverage to about 95%. It needs its own accuracy check. GDN projections feed a recurrent state, where error can accumulate. That is a plausible reason RedHat excluded them, though the card doesn't say.
- **Overheads to watch:**
  - Dynamic per-token activation quantization adds a quant kernel per GEMM input unless fused (`fuse_norm_quant`/`fuse_act_quant` at -O2).
  - The MTP head stays BF16, which is fine.
  - Smaller weights (about 14 GB vs 18 GB) free about 4 GB for KV/state. That helps admission at peak.
- **Accuracy risk:** the base-model numbers show ≤0.6% loss, but your AWQ experience showed the agent class is the sensitive one (2/20 vs 3-7/20). Validate W8A8 MANGO on the agent class with multiple seeds before any official run.
- **Steps to produce it (rule-compliant):**
  1. Run llm-compressor (transformers ≥5) on CMKL/MANGO1.5-Qwen3.5-9B with GPTQ W8A8, dynamic per-token activations, `dampening_frac=0.01`, the ignore list above, and optionally SmoothQuant 0.8. Calibrate on about 512 in-domain samples (task-like prompts, including long reasoning and agent traces).
  2. Copy the MTP tensors back.
  3. Serve with the same flags. vLLM auto-detects `compressed-tensors`. Check the log for `CutlassInt8ScaledMMLinearKernel`.
  4. Benchmark peak TPOT with the local final track.
- **GPTQ/Machete/Exllama on A100:** Machete targets Hopper. Exllama and Marlin are weight-only (A16), so they hit the same compute-bound problem at batch. They are not recommended. This is an inference consistent with your AWQ/Marlin results. `VLLM_DISABLED_KERNELS` can exclude kernels (Machete, Marlin, Exllama) if needed (local vLLM 0.32.0 source, `envs.py` ~1220).

### Gaps
- I found **no primary A100 benchmark of 8B-class W8A8-INT8 vs BF16 at batch 64-256 in recent vLLM**. The closest are A6000 8B (1.38x async) and 4x A100 70B (1.87x), both on older vLLM. A local microbenchmark is needed (e.g., `vllm bench throughput` with both checkpoints).
- I found no data on W8A8 combined with GDN hybrids and MTP acceptance. Quantization can lower MTP acceptance if the target distribution shifts, so measure acceptance rates.
- I didn't verify whether the RedHat recipe used SmoothQuant (the card mentions only GPTQ).

## Energy: static clock/power for LLM decode on A100

### Takeaway
The literature's finding that decode is frequency-insensitive and memory-bound holds for **small or moderate batches**. At high load, savings collapse:
- GreenLLM's decode-energy savings fall from 20-25% to 8-12% at high TPS on A100.
- Your E023 showed about 1:1 SLO loss with energy at 1215 MHz.

This is consistent with peak steps being compute-bound. A static clock lock is not a free energy win for this workload. The energy lever is doing fewer FLOPs or cheaper FLOPs (INT8, dynamic MTP depth), not a lower clock.

### Cited Findings
- **Maliakel et al. (1B-32B models).** Decode is 77-91% of inference time and "largely insensitive to GPU frequency". Cutting from 2842 MHz to 180 MHz saved about 42% energy for a 1-6% latency increase. The GPU and batch regime differ from yours (not an A100 at batch 130+). Source: [arXiv 2501.08219](https://arxiv.org/pdf/2501.08219).
- **GreenLLM** (Qwen3-14B/30B BF16, **A100-SXM4-40GB**, Dynamo + TRT-LLM, NVML application clocks):
  - Prefill energy is U-shaped, with the optimum at about 0.95-1.05 GHz (70-80% of the 1.41 GHz max).
  - The decode optimum is lower than prefill's. Its dynamic controller moved decode clocks from about 450 MHz to about 1.35 GHz as TPS rose.
  - Decode-energy savings: 20-25% at low TPS, 8-12% at 2400-3000 TPS (p99 TBT 83.2 vs 84.6 ms default).
  - Trace savings: 27.5% at 1 QPS, falling to 6.8% at 10 QPS on Alibaba chat.
  - The default NVIDIA governor kept clocks at 1.1-1.4 GHz regardless of load.
  - Source: [arXiv 2508.16449](https://arxiv.org/html/2508.16449v1).
- **throttLL'eM:** iteration-level frequency and instance scaling driven by predicted KV usage and batch size. Up to 43.8% lower energy and ≥1.71x energy efficiency under SLOs, versus NVIDIA Triton server. It is dynamic, so it conflicts with your "constant for the whole run" rule. Source: [arXiv 2408.05235](https://arxiv.org/html/2408.05235v2).
- **DynamoLLM:** cluster-level reconfiguration (instances, TP, frequency) saves 53% energy and 38% carbon under SLOs. It is multi-GPU and dynamic, so not applicable. Source: [arXiv 2408.00741](https://arxiv.org/html/2408.00741v1).
- **Your own results** (local `~/repo/KEY_FINDINGS.md`):
  - Clock lock 1215 MHz: J/req -25%, on-time -24%. Official final 52.5 vs 59.0.
  - Power cap 250/300 W: J/req -13..-18%, prefill slows and TTFT fails rise.

### Inferences
- Your measured average power is 345-370 W at a 400 W limit, so the GPU is mostly not power-throttled at default. It runs near max boost (about 1.41 GHz) during compute-bound peaks.
  - A static lock just below max (e.g., 1305-1350 MHz, about 5-7% lower) would save only a few percent of energy and would cost peak SLO proportionally. Low expected value.
  - Given the "constant for the whole run" rule, a static clock can't exploit off-peak memory-bound phases the way GreenLLM or throttLL'eM do.
- Off-peak energy is dominated by idle/low-utilization power and MTP waste. Dynamic MTP depth (k=2 off-peak) and INT8 (fewer and cheaper MMA cycles) cut joules without cutting SLO capacity. Per the framing above, these are the energy levers consistent with your data.

### Gaps
- I found no published A100 static-clock sweep at batch 100-200 with speculative decoding on a GDN hybrid.
- ML.ENERGY leaderboard data and "Characterizing Power Management Opportunities for LLMs in the Cloud" (POLCA) were not retrieved in this pass.

## Other: max-num-seqs vs CUDA graph sizes, chunked prefill in v1, OS/CPU knobs

### Takeaway
- `max-num-seqs 256` with k=2 exceeds full-CUDA-graph coverage (about 170 seqs at capture 512). This is harmless at your 130-140 peak.
- Chunked prefill in v1 is always on, and your budget knobs (4096 budget, 1024 per-request cap) were already tuned empirically.
- OS/CPU knobs are unresearched and likely second-order. Avoid `--stream-interval`.

### Cited Findings
- **Full-graph decode coverage** = `max_num_seqs * (1 + num_speculative_tokens)` tokens, bounded by `max_cudagraph_capture_size` (default 512). Source: local vLLM 0.32.0 source, `config/vllm.py` (~660-685).
- **Async scheduling** is already auto-enabled for MTP (it overlaps CPU scheduling with GPU execution). Source: local logs ("Async scheduling is already on (mtp is in EagleModelTypes)").
- **High-concurrency frontend knobs** in the vLLM Qwen3.5 recipe: `--stream-interval 100` (buffers 100 tokens) and `--api-server-count 1` used only to restore stats logging. Source: [vLLM blog](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps).
- **Chunk-size history** (your notes): MAX_BT 4096 plus a 1024 per-request prefill cap is the best TTFT config. 2048/512 caps and a 2048 budget were worse. Source: local `~/repo/KEY_FINDINGS.md` (E017/E019).

### Inferences
- At peak, TPOT is set by the per-step prefill chunk compute (up to 4096 tokens, about 4-5x the decode verify tokens) more than by decode itself. Two levers are therefore coupled through the prefill-decode trade-off:
  - TPOT can be traded against TTFT only via the budget and caps (already tuned).
  - Raising per-step throughput (INT8, k=1 at peak) improves both TPOT and TTFT.
- If you set `max-num-seqs` above about 170 with k=2, raise `max_cudagraph_capture_size` to `3 x max_num_seqs` (e.g., `--compilation-config '{"max_cudagraph_capture_size":768}'`) so pure-decode bursts stay in full graphs. It is irrelevant at your current peak.

### Gaps
- I found no sourced evidence on process priority (`nice`/`chrt`), CPU affinity (`taskset`/numactl), uvloop, or `--api-server-count` effects on TTFT jitter for single-GPU vLLM 0.32. Not researched; likely second-order given async scheduling and GPU-bound steps.
