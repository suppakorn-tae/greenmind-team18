# Speculative decoding on a hybrid Qwen3.5 (GDN) model under high concurrency: latency vs. memory levers (vLLM 0.32.0, 1x A100-40GB)

Source convention: "LOCAL" means the installed vLLM 0.32.0 source at `/home/researcher/vllm-env/lib/python3.12/site-packages/vllm/` (path relative to that root, with the line number), read directly on 2026-10-24. "REPO" means our own experiment logs under `/home/researcher/repo/experiments/`. Each lever is tagged by availability: **[0.32 as shipped]**, **[open PR: needs a patch]**, **[third-party weights]**.

**The constraint behind most answers below.** In 0.32.0, the number of GDN (Mamba-type) state slots each running request holds is fixed at startup from the *static* `num_speculative_tokens`:
- `MambaSpec(num_speculative_blocks = 0 if use_replayssm else vllm_config.num_speculative_tokens)`. LOCAL `model_executor/layers/mamba/abstract.py` L79-83.
- Per-request memory is `page_size_bytes * (1 + num_speculative_blocks)` in mamba_cache_mode "none". LOCAL `v1/kv_cache_interface.py` L1075-1082.
- The manager always allocates the extra `num_speculative_blocks` blocks per request. LOCAL `v1/core/single_type_kv_cache_manager.py` L1698-1702 and L1785-1788.
- Nothing in that path reads the per-step dynamic K. So **any drafter (MTP, n-gram, EAGLE, DFlash, suffix) with K drafts costs (K+1) GDN slots per sequence**, and a dynamic-K schedule costs slots for its *maximum* K.

---

## Q1. Dynamic / adaptive speculation length in vLLM (config keys, MTP + hybrid support)

### Takeaway
vLLM 0.32.0 ships batch-size-scheduled dynamic K: `num_speculative_tokens_per_batch_size: [[start_bs, end_bs, K], ...]` inside `--speculative-config`. Our MTP path appears to support it under Model Runner V2, which we already run. It does **not** reduce GDN state memory, though, because slots are sized by the static max K. So "k=3 at low load, k=2 at peak" keeps E020/E032's k=3 memory footprint and its final-track collapse. The memory-neutral use is max K=2 with a step-down to K=1 at peak. The older `speculative_disable_by_batch_size` / `disable_by_batch_size` keys do not exist in 0.32's SpeculativeConfig. Goodput-based (TurboSpec-style) automatic K is not in vLLM; only manual schedules are.

### Cited Findings
- **Config key [0.32 as shipped]:** `num_speculative_tokens_per_batch_size: list[tuple[int, int, int]] | None`, documented as "Batch-size schedule used to dynamically choose speculative-token count. Each entry is (range_start, range_end, num_speculative_tokens) with an inclusive batch-size range." — LOCAL `config/speculative.py` L490-496.
- **Schedule rules:**
  - The first range must start at 1. Ranges must be non-overlapping and sorted, and K may be 0.
  - Gaps carry the previous K forward, and the tail carries the last K up to `max_num_seqs`.
  - Every value is clamped to `min(static num_speculative_tokens, K)`, so the static `num_speculative_tokens` must be set to the schedule's maximum.
  — LOCAL `v1/spec_decode/dynamic/utils.py` L7-150.
- **How K is chosen at runtime:** the scheduler looks up `dynamic_sd_lookup[len(num_scheduled_tokens)]` each step. "Batch size" is therefore the number of requests scheduled in that step, including requests that are mid-prefill-chunk. — LOCAL `v1/core/sched/scheduler.py` L1485-1490 (lookup built at L307-314).
- **Side effect:** when dynamic SD is on, the scheduler stops padding decode requests to `1 + num_spec_tokens`. — LOCAL `v1/core/sched/scheduler.py` L1135.
- **CUDA graphs and data parallelism:**
  - On Model Runner V1, full CUDA graphs are overridden to PIECEWISE when dynamic SD is on. "Use VLLM_USE_V2_MODEL_RUNNER=1 if you want to use full CUDA graphs." — LOCAL `config/vllm.py` L1039-1056.
  - With DP > 1, dynamic SD is disabled automatically. — LOCAL `config/vllm.py` L1058-1075.
  - V2 captures per-tier decode graphs (`specialize_spec_tokens=... uses_dynamic_speculative_decoding()`) in the target-dependent AR speculator, the base class of `MTPSpeculator`. — LOCAL `v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py` L145-161; `v1/worker/gpu/spec_decode/mtp/speculator.py` L7-12; tier capture sizes in `config/vllm.py` L2396-2434.
- **We already run V2:** our server logs "Using V2 Model Runner" (REPO `experiments/E040/server.log`). — REPO.
- **Official docs:**
  - Dynamic SD "adjusts the number of draft tokens (K) per decoding step based on batch size". Example: `"num_speculative_tokens": 3, "num_speculative_tokens_per_batch_size": [[1,64,3],[65,128,1],[129,512,0]]`.
  - EAGLE-3 example schedule: `[1,16,5],[17,32,4],[33,64,3],[65,128,1],[129,512,0]`.
  - Tested methods: EAGLE, EAGLE-3 and DFlash. "Other SD methods may or may not work out of the box." MTP is not named.
  - No benchmarks and no schedule-generation tool are given. K is a manual schedule.
  — [vLLM docs: Dynamic Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)
- **Open bug-fix PRs touching dynamic SD with AR/MTP-style speculators** (these show rough edges):
  - #48329 "Skip Dynamic SD schedule for autoregressive speculator decode CUDA graphs".
  - #58821 "Avoid dynamic query lengths for draft decode graphs" (ZeroDivisionError).
  - #56136 "Ladder cudagraph candidates per uniform query length": FULL-graph coverage rose from 101/128 to 128/128 batches for one schedule.
  - #62040 "Keep a full uniform-decode batch reachable by FULL graphs": +4.4% decode throughput on GB200.
  - #46399 "Disable dynamic SD for fixed-K proposers instead of crashing" (merged 2026-10-17).
  — [GitHub search results, vllm-project/vllm](https://api.github.com/search/issues?q=repo:vllm-project/vllm+%22dynamic+speculative+decoding%22+is:pr&per_page=15)
- **Reported dynamic-K gain (DSpark, 4x H100, HumanEval, batch 64):** fixed K=16 about 381 tok/s, K=6 schedule about 526 tok/s, K=0 about 882 tok/s. At high batch, less speculation won. — [PR #47737](https://github.com/vllm-project/vllm/pull/47737) (via GitHub search API summary)
- **Other adaptive features in 0.32:**
  - `enable_adaptive_verification` is "Currently only supported for method='dspark'". — LOCAL `config/speculative.py` L551-553.
  - Docs list Adaptive Verification as "currently DSpark only". — [vLLM spec decode docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
  - An open PR extends adaptive DSpark verification to ragged Qwen GDN batches. — [PR #53929](https://github.com/vllm-project/vllm/pull/53929)
  - Sequence-length-based schedules (`num_speculative_tokens_per_seq_len`) and a per-request cap (`max_speculative_tokens`) are open, unmerged PRs. — [#54801](https://github.com/vllm-project/vllm/pull/54801), [#63151](https://github.com/vllm-project/vllm/pull/63151)
- **TurboSpec (formerly SmartSpec), Liu et al., Berkeley/vLLM team, v3 July 2025:** profiles the system and uses a feedback loop to choose the speculation amount that maximizes predicted goodput, implemented on vLLM. I extracted no numbers (abstract only). — [arXiv 2406.14066](https://arxiv.org/abs/2406.14066)
- **DSDE:** per-sequence dynamic lookahead in a modified vLLM 0.8.4. It ran in eager mode because changing K would require recapturing CUDA graphs. — [arXiv 2509.01083](https://arxiv.org/pdf/2509.01083)

### Inferences
- **Candidate config A (memory-neutral, [0.32 as shipped]):**
  ```
  --speculative-config '{"model":"CMKL/MANGO1.5-Qwen3.5-9B","method":"mtp","num_speculative_tokens":2,"num_speculative_tokens_per_batch_size":[[1,96,2],[97,256,1]]}'
  ```
  - Memory stays as today (k=2 slots). Verification compute drops only in peak steps.
  - KEY_FINDINGS says k=1 costs +23% J/req globally, so step down only at the very top. Sweep the threshold (e.g. 96/112/128).
  - Batch size counts prefill-chunk requests too. The threshold therefore trips at somewhat lower decode concurrency than the "peak seqs" metric suggests.
- **Candidate config B (TPOT at low/mid load, does NOT fix memory):** `num_speculative_tokens:3` with `[[1,64,3],[65,256,2]]`.
  - This keeps k=3 GDN slots for every sequence, so the final-track admission collapse seen in E020/E032 should recur.
  - Only worth testing together with a memory fix (Q2: RecoverSSM).
- **Expected value of stepping down:** k=2 gives about 2.61 expected tokens per step (1+0.87+0.74) for 3 verified tokens/seq. k=1 gives about 1.87 for 2 tokens/seq.
  - At 130+ seqs on an A100 (ridge of roughly 200 tokens/step for a 9B BF16 model), steps are compute-bound. Step time then scales roughly with verified tokens, so K=1 at peak could be TPOT-neutral or better.
  - This is a roofline estimate, not measured.
- **Correctness risk:** GDN metadata builds spec state indices as `block_table[:, :num_spec+1]` from the static `num_spec`, with per-request draft counts from `num_decode_draft_tokens_cpu`. LOCAL `v1/attention/backends/gdn_attn.py` L121-125, L288-300, L381-398.
  - Whether a batch-wide K below the static max works correctly on GDN in 0.32 is unverified. PR #53929, which adds ragged GDN spec tests, suggests the area is still in progress.
  - Run a greedy output-identity smoke test against static k=2 before benchmarking.

### Gaps
- No vLLM benchmark of dynamic SD with MTP, with a hybrid/GDN model, or on A100.
- No TurboSpec latency numbers extracted (abstract only).
- No source confirms whether 0.32.0 includes #46399's behavior, or whether MTP counts as a "fixed-K proposer" there.
- Not tested whether K=0 tiers work with GDN.

---

## Q2. How vLLM allocates GDN state for spec decode, and how to shrink it so k=3 fits

### Takeaway
Each running sequence holds (1+K) full GDN recurrent-state copies across all 24 GDN layers. For this model that is about 27 MiB per slot after page padding: about 81 MiB/seq at k=2 and 108 MiB/seq at k=3. At 130 seqs that is about 10.3 GiB vs 13.7 GiB, out of a 17.4 GiB pool.

0.32.0 has no shipped option to avoid the per-draft slots for Qwen3.5:
- `--use-replayssm` sets `num_speculative_blocks=0`, but 0.32 allows it only for architectures declaring `SupportsReplaySSM` (Nemotron-H, Kimi KDA).
- The Qwen3.5/Qwen3-Next port, "RecoverSSM", is an unmerged draft PR stack (#58863 + #61309, optionally #62625).
- Mamba state dtypes go no lower than 16-bit, and `mamba_cache_mode` only offers `none`/`align`.

### Cited Findings
- **Slot count = 1 + K (shipped Mamba/GDN spec path)** — LOCAL `model_executor/layers/mamba/abstract.py` L79-83; `v1/kv_cache_interface.py` L1075-1082; `v1/core/single_type_kv_cache_manager.py` L1698-1702.
- **GDN metadata side:** keeps a `spec_state_indices_tensor` of shape `[batch, num_spec+1]`, plus decode CUDA-graph buffers of `max_num_seqs*(num_spec+1)`. — LOCAL `v1/attention/backends/gdn_attn.py` L64, L121-125, L145-174.
- **Model dimensions (MANGO config):**
  - 32 layers with `full_attention_interval` 4, i.e. 8 attention + 24 linear_attention layers.
  - GDN: `linear_num_value_heads` 32, `linear_key_head_dim`/`linear_value_head_dim` 128, `linear_num_key_heads` 16, `linear_conv_kernel_dim` 4.
  - Attention: `num_key_value_heads` 4, `head_dim` 256.
  - MTP: `mtp_num_hidden_layers` 1.
  — LOCAL file `/home/researcher/models/MANGO1.5-Qwen3.5-9B/config.json`.
- **Our logs (REPO):**
  - Champion (k=2): "Setting attention block size to 288 tokens to ensure that attention page size is >= mamba page size", "Add 3 padding layers, may waste at most 12.50% KV cache memory", "Available KV cache memory: 17.36 GiB", "GPU KV cache size: 475,367 tokens".
  - The no-spec run E004 logs no padding warning.
  — REPO `experiments/champion/server.log`, `experiments/E004/server.log`.
- **The profiled KV pool swings by about 3.9 GiB between runs with the same vLLM flags (REPO):**
  - E020 vs E032: both k=3 with the same vLLM serve flags (E032 adds only the thinking-budget wrapper). Pools were 13.55 GiB (peak activation 2.99 GiB) vs 17.43 GiB (1.31 GiB).
  - E017 vs E020: same flags except k. Pools were 17.36 GiB vs 13.55 GiB.
  - Champion (k=2): 17.36 GiB, activation 1.38 GiB.
  - The log suggests pinning with `--kv-cache-memory=<bytes>`.
  — REPO `experiments/{E017,E020,E032,champion}/server.log`.
- **Peak occupancy in practice (REPO):**
  - E032 (k=3, 17.43 GiB pool): 107-115 running at 98.5-100% KV usage with 6-28 waiting. Earlier in the run, 79 running was at 72%.
  - E033 (k=2, 18.94 GiB pool): 136-145 running at 95.2-98.5%.
  - So k=2 is also close to memory saturation at our peak.
  — REPO `experiments/E032/server.log`, `experiments/E033/server.log` ("Running: N reqs ... GPU KV cache usage" lines).
- **Options available in 0.32:**
  - `MambaDType = Literal["auto","float32","float16","bfloat16"]`: no FP8/INT8 state.
  - `MambaCacheMode = Literal["align","none"]`.
  - `--use-replayssm`: "cache recent SSM inputs and skip the per-step full-state store... Mamba2 speculative decode requires FlashInfer and mamba_cache_mode 'none'".
  — LOCAL `config/cache.py` L62-63, L181-216.
- **ReplaySSM gating in 0.32:** rejected unless `model_config.supports_replayssm`. Only `nemotron_h.py` (and Kimi K3 KDA "RecoverSSM") declare it. — LOCAL `config/vllm.py` L3318-3420; `grep supports_replayssm model_executor/models/` returns only `nemotron_h.py`.
- **RecoverSSM for Qwen3.5 / Qwen3-Next GDN [open PR: needs a patch]:**
  - What it does:
    - GDN verifies the draft window from a read-only checkpoint and stores a per-token fp32 replay record.
    - After sampling, it replays only the accepted tokens, so `num_speculative_blocks = 0` (no per-draft blocks).
    - The record grows the Mamba page; on Qwen3.5-4B the attention block goes 288 to 320 tokens at K=3 and 352 at K=5.
  - Flags (tested): `VLLM_USE_V2_MODEL_RUNNER=1 ... --speculative-config '{"method":"mtp","num_speculative_tokens":5}' --enable-prefix-caching --mamba-cache-mode align --mamba-cache-dtype bfloat16 --mamba-ssm-cache-dtype bfloat16 --mamba-backend triton --attention-backend TRITON_ATTN --use-replayssm`.
  - Results (1x L40S, Qwen3.5-4B, c=32): +8.6% output tok/s at K=5 and +3.4% (noisy) at K=3. Acceptance length unchanged (3.93 at K=5, 3.18 at K=3). GSM8K 0.773 vs 0.771.
  - Greedy outputs diverge on 27/30 prompts, at bf16 near-ties.
  - Not tested on Ampere or in mamba_cache_mode "none".
  - Status: draft, needs-rebase.
  — [PR #61309](https://github.com/vllm-project/vllm/pull/61309)
- **Base PR #58863 (Qwen4Exp, open, needs-rebase):**
  - "KV tokens in the same 4 GiB: 75,678 to 103,953 (+37%)".
  - A maintainer reported up to +17% output tok/s at high concurrency, and H200 preemptions at C64 went from 188 to 0.
  - Prefix hit rate fell 0.489 to 0.381 in one commenter's setup because of the larger block.
  — [PR #58863](https://github.com/vllm-project/vllm/pull/58863)
- **PR #62625 (draft):** moves the replay records out of the Mamba page into a per-request-slot buffer, restoring the stock block size. About 452 MiB at 100 seqs, K=5, 24 GDN layers. — [PR #62625](https://github.com/vllm-project/vllm/pull/62625) (via GitHub search API summary)
- **Related Qwen3.5 MTP memory PR:** #54201 avoids temporary full-vocab MTP embedding/LM-head allocations. This is a load-time peak only (about 4.85 GiB on a 27B INT4); steady-state model memory is unchanged. — [PR #54201](https://github.com/vllm-project/vllm/pull/54201) (via GitHub search API summary)

### Inferences
- **Per-sequence arithmetic** (inference; check against logs):
  - SSM state per GDN layer: 32 x 128 x 128 x 2 B = 1.00 MiB. Conv state: (2x16x128 + 32x128) x 3 x 2 B = 48 KiB. Unpadded total about 1.047 MiB per layer per slot.
  - The log's 288-token attention block implies a unified page of 288 x 2 (K,V) x 4 heads x 256 x 2 B = 1.125 MiB per layer.
  - With padding, one slot is 24 x 1.125 = 27 MiB/seq. So k=2 is 81 MiB/seq, k=3 is 108 MiB/seq, and each extra K costs 27 MiB/seq.
  - At 130 seqs on the 17.36 GiB pool: k=2 needs about 10.3 GiB (59%) and k=3 about 13.7 GiB (79%) just for GDN state.
  - That leaves about 7.1 GiB vs 3.6 GiB for attention KV. At about 36-48 KiB/token (9 attention layers incl. MTP, plus padding) that is roughly 150-200K vs 80-100K tokens, i.e. about 1.2-1.6K vs 0.6-0.8K tokens per sequence.
  - That explains why k=3 admission stalls at about 130 seqs with 18-20K-token prompts in flight.
- **"3 padding layers" is caused by the MTP draft's own attention layer.** 8 target + 1 MTP = 9 attention layers, against 24 GDN layers grouped 12 per group. E004 without MTP had no padding.
  - vLLM bounds the waste at ≤12.5% of the pool, roughly 2 GiB. That is roughly the k=3 increment for about 75 seqs.
  - No 0.32 flag removes it.
- **Sanity check of the per-seq figure:** E032's peak (about 115 seqs filling a 17.43 GiB pool) implies about 155 MiB/seq total at k=3. E033's (about 145 seqs in 18.94 GiB) implies about 134 MiB/seq at k=2. The k=3 GDN state estimate (108 MiB) is the majority of that, and the observed k=3-k=2 gap (about 21 MiB) is near the predicted 27 MiB, given different contexts.
- **Cheapest shipped mitigation:** pin the KV pool with `--kv-cache-memory` at the good-profile size (≈17.36 GiB ≈ 18.64e9 bytes), and validate on the heavy track.
  - This removes the ~3.9 GiB run-to-run swing (E020 vs E032). A 13.5 GiB profile means about 25-30 fewer concurrent sequences at k=2.
  - Past experiment comparisons should be read alongside each run's "Available KV cache memory" line.
  - Going higher (the log's "fully utilize" figure of about 19.3 GiB) risks OOM, because some profiles measured 3.0-3.2 GiB peak activation and the cause of the variance is unknown.
  - It cannot make k=3 fit by itself. E032 had 17.43 GiB and still collapsed. k=3 at 130 seqs needs about 3.4 GiB more than k=2.
- **Practical path to "k=3 that fits":** backport RecoverSSM (#58863 + #61309, ideally + #62625) onto 0.32.0.
  - Expected state memory: about 27 MiB/seq plus a fp32 replay record of a few MiB/seq. #62625 measured about 4.5 MiB/seq at K=5 (452 MiB / 100 seqs); the shape-based K=3 estimate is about 1.6 MiB.
  - So about 30 MiB/seq at k=3 vs today's 81 MiB at k=2, freeing about 6 GiB at 130 seqs.
  - Risks: 18+ commit unmerged stack; tested only with V2 + `--mamba-backend triton` + align mode on L40S/H200/GB10, not A100 and not mode "none"; greedy outputs not bit-identical to stock; needs a re-check of the task-accuracy gate.
  - Check whether the event rules allow a patched vLLM.
- **Without patches, the k=3 memory trade cannot be bought in 0.32.** The levers are: (a) keep k=2 and pin `--kv-cache-memory`; (b) dynamic step-down at peak (Q1, config A); (c) free other memory, outside this topic: FP8 attention KV on A100, smaller `--max-model-len` if the workload allows. A smaller max-model-len does not shrink GDN state.

### Gaps
- No GitHub issue found that specifically reports Qwen3.5/Qwen3-Next MTP admission/memory collapse at high concurrency. Ours appears to be undocumented upstream.
- Unknown whether RecoverSSM supports `mamba_cache_mode none` (our setting) or Ampere's Triton paths.
- Unknown whether vLLM shares the GDN conv-state slot differently from the SSM state. I assumed both are replicated per spec slot.
- Exact token capacity per block group not reconciled against the logged "475,367 tokens" figure.
- Cause of the profiled peak-activation variance (1.3 vs 3.0 GiB with the same flags) not investigated.

---

## Q3. Alternatives and complements: suffix, n-gram / prompt-lookup, EAGLE-3, DFlash, MTP + n-gram

### Takeaway
None of the alternative drafters fixes the binding constraint, because every drafter pays (K+1) GDN slots per sequence. Several make it worse:
- DFlash wants K≈7-15.
- Suffix decoding defaults K to 24 and is V1-runner-only, requiring arctic-inference (not installed).
- n-gram typically uses K=4-5.

Community EAGLE-3 heads for Qwen3.5-9B report acceptance length about 2.97 at 3 steps, no better than our built-in MTP, and they add draft KV and weights. Combining MTP with n-gram in one server is not supported in 0.32 (single `method`).

### Cited Findings
- **Methods in 0.32:** `ngram`, `ngram_gpu`, `suffix`, `draft_model`, `medusa`, `mlp_speculator`, `custom_class`, `eagle`, `eagle3`, `dflash`, `dspark` and the MTP family (incl. `qwen3_5_mtp`, `qwen3_next_mtp`, `mtp`). The speculative config has one `method` field. — LOCAL `config/speculative.py` L38-82, L399.
- **N-gram:**
  - `prompt_lookup_min`/`max` default to 5 if both are unset. — LOCAL `config/speculative.py` L1197-1220.
  - Docs example: `{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_max":5,"prompt_lookup_min":2}`. — [vLLM spec decode docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/)
- **Suffix decoding:**
  - Requires Arctic Inference.
  - `num_speculative_tokens` defaults to `suffix_decoding_max_tree_depth` (24).
  - Knobs: `suffix_decoding_max_cached_requests` 10000, `suffix_decoding_max_spec_factor` 1.0, `suffix_decoding_min_token_prob` 0.1.
  — LOCAL `config/speculative.py` L505-523, L1596-1629.
  - V2 runner lists `suffix` (and medusa, mlp_speculator, custom_class) as unsupported, so vLLM falls back to V1. — LOCAL `config/vllm.py` L2914-2932.
  - `pip list` in our venv shows no arctic-inference. — REPO environment check.
- **SuffixDecoding paper (NeurIPS 2025 spotlight):** up to 5.3x on agentic benchmarks (SWE-Bench, Text-to-SQL), about 2.8x faster than EAGLE-2/3 and about 1.9x faster than Token Recycling. Code is in snowflakedb/ArcticInference. — [arXiv 2411.04975](https://arxiv.org/abs/2411.04975)
- **"Speculative Decoding: Performance or Illusion?" (Liu et al., Berkeley; H100, vLLM 0.10.1.1):**
  - On Qwen3-8B-Thinking (batch 1-16), EAGLE-3 gives 1.64-1.80x and n-gram 1.50-1.58x on GPQA/AIME.
  - On code editing (InstructCoder), n-gram beats EAGLE/EAGLE-3: mean generated length 7.27 vs 4.24. When prompt-output BLEU-4 > ~0.6, n-gram wins at all batch sizes, by up to 53% more speedup at K=3 and up to 100% at K=5.
  - On GSM8K, n-gram's mean generated length is 1.41.
  - Verification is 42-95% of execution time, growing with batch. EAGLE drafting is 12-20% at bs 1 and 3-7% at bs 512. n-gram drafting is <2%.
  — [arXiv 2601.11580v2](https://arxiv.org/html/2601.11580v2)
- **EAGLE-3 heads for Qwen3.5-9B [third-party weights]:** `yuyijiong/Qwen3.5-9B-Eagle3` (Apache-2.0, 0.5B, trained on open-perfectblend).
  - SGLang, temp 0, mean of MT-Bench/GSM8K/HumanEval, 512 max tokens: acceptance length/speedup 2.97/2.25x at 3 steps, 3.97/2.59x at 7 steps, 4.34/2.19x at 15 steps.
  - At temp 1.0: 2.34/1.74x at 3 steps.
  - Thinking-mode training not stated. No vLLM config given.
  — [HF model card](https://huggingface.co/yuyijiong/Qwen3.5-9B-Eagle3)
  - Other HF repos: `BLR2/Eagle3-Qwen3.5-9B`, `dobriak/Eagle3-Qwen3.5-9B`. — [HF API search](https://huggingface.co/api/models?search=Qwen3.5-9B-eagle&limit=10)
  - The RedHatAI speculators collection lists EAGLE-3 for Qwen3-8B etc., not Qwen3.5-9B. — [vLLM speculators docs](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/algorithms/eagle3/)
- **DFlash for Qwen3.5-9B [third-party weights]:** `z-lab/Qwen3.5-9B-DFlash` (Apache-2.0, about 1B params, trained with 40k seq length + sliding window).
  - B200, greedy, thinking on: speedup 3.1-5.0x at c=1 and 1.55-2.58x at c=32. Block 8 is best at c=32 and recommended for high concurrency.
  - Acceptance length 4.6-5.8 (block 8) and 5.7-7.9 (block 16) at c=1.
  - The card says vLLM support is in PR #40898 "not yet merge-ready". `method: "dflash"` does exist in 0.32's method list.
  — [HF model card](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash); LOCAL `config/speculative.py` L68.
- **MTP + n-gram hybrid:** no hybrid method found in 0.32 config. Arctic Inference's suffix decoding is its own V1-only path. — LOCAL `config/speculative.py` (single `method`); I found no vLLM source for combining them.

### Inferences
- DFlash with block 8 means about 8 GDN state slots per sequence (8 x 27 MiB ≈ 216 MiB/seq). At 130 seqs that is about 28 GiB, more than the whole pool, so DFlash is infeasible on 40 GB at our concurrency unless RecoverSSM-style state handling lands. It also adds about 2 GB of draft weights and draft KV. Ignore for now.
- **EAGLE-3:**
  - Its acceptance (2.97 at 3 steps, on base Qwen3.5-9B, short outputs) is no better than our MTP's 3.13 at k=3 (1+0.84+0.70+0.59).
  - It was trained on the official base, not the MANGO fine-tune, so acceptance on MANGO would likely be lower.
  - It adds draft attention KV per token plus 0.5B weights. Not worth pursuing.
  - Training our own EAGLE-3/MTP fine-tune on MANGO outputs is allowed only if "derived from official weights during the event". Organizers must confirm.
- **n-gram:**
  - Pays the same GDN slots at K=4-5 for no model-based fallback. On thinking-mode text, acceptance is heavy-tailed and lower (paper: 1.41 mean on GSM8K).
  - It would only win on agent/RAG turns whose answers copy the prompt. Since `method` is global, it would hurt the reasoning classes.
  - Not recommended unless a per-request method switch existed (it does not in 0.32).
- `rejection_sample_method: "block"`: under greedy (temp 0) verification, joint block verification should accept the same tokens as standard greedy matching (inference), so no gain is expected.
- Excluded: FLy verification (lossy), `synthetic` rejection (benchmarking only).

### Gaps
- No measured acceptance of any third-party Qwen3.5-9B drafter on the MANGO derivative or on our workload.
- No vLLM high-batch numbers for EAGLE-3 or n-gram on hybrid models.
- Unknown whether DFlash PR #40898 is fully merged in 0.32. The method exists, but the card says otherwise.

---

## Q4. Typical MTP acceptance for Qwen3-Next / Qwen3.5; temp-0 and thinking effects; draft TP / quantization

### Takeaway
Our per-position acceptance (0.87/0.74 at k=2; 0.84/0.70/0.59 at k=3, i.e. acceptance length about 3.1) matches or beats upstream reports. Qwen3.5-4B MTP shows acceptance length about 3.18 at K=3 and about 3.93 at K=5. Qwen3.5-122B shows about 78% at k=2 and 86% at k=1. Temp 0 helps acceptance relative to sampling. Draft TP and draft quantization knobs don't apply to a built-in MTP head on 1 GPU.

### Cited Findings
- **Qwen3.5-4B native MTP (L40S, spec_bench, c=32):** acceptance length 3.177-3.181 at K=3 and 3.931-3.940 at K=5. GSM8K K=5 acceptance length 4.19. — [PR #61309](https://github.com/vllm-project/vllm/pull/61309)
- **Qwen3.5-122B-A10B, 8x A100-80GB, `qwen3_next_mtp`:** draft acceptance about 78% at k=2 and 86% at k=1. — [PR #51769](https://github.com/vllm-project/vllm/pull/51769) (via GitHub search API summary)
- **Qwen3.5-family 27B INT4 MTP:** 116/137 draft tokens accepted (84.7%) in a small sample. — [PR #54201](https://github.com/vllm-project/vllm/pull/54201) (via GitHub search API summary)
- **Low-acceptance outlier:** nvidia/Qwen3.6-27B-NVFP4, k=3, vLLM 0.25.1: per-position 0.353/0.333/0.294 (mean length 1.98), in a bug report with structured outputs. — [Issue #49210](https://github.com/vllm-project/vllm/issues/49210)
- **Official vLLM Qwen3-Next recipe:** uses `{"method":"qwen3_next_mtp","num_speculative_tokens":2}`. — [vLLM recipes: Qwen3-Next](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html)
- **GLM-4.5-Air MTP (reused single module):** position-wise acceptance 0.92 → 0.68 → 0.38. — [arXiv 2601.11580v2](https://arxiv.org/html/2601.11580v2)
- **Temperature effect (EAGLE-3 on Qwen3.5-9B, 3 steps):** acceptance length 2.97 at temp 0 vs 2.34 at temp 1.0. — [HF yuyijiong/Qwen3.5-9B-Eagle3](https://huggingface.co/yuyijiong/Qwen3.5-9B-Eagle3)
- **Thinking mode:** z-lab's DFlash benchmarks used greedy with thinking enabled and still reached acceptance length 4.6-7.9. — [HF z-lab/Qwen3.5-9B-DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash)
- **Default draft sampling is greedy.** `draft_sample_method` "greedy" treats draft probs as one-hot. "probabilistic" costs extra GPU memory. — LOCAL `config/speculative.py` L600-606.
- **Draft TP and quantization:**
  - `draft_tensor_parallel_size` "Can only be 1 or the same as the target model's tensor parallel size".
  - `quantization` for the draft "only takes effect when using the draft model-based speculative method".
  — LOCAL `config/speculative.py` L407-418.

### Inferences
- Our MTP head is already near the top of reported Qwen3.5 acceptance. Gains must come from memory and scheduling, not a better drafter.
- At temp 0 with thinking, k=3 adds about 0.59 tokens/step (+20% tokens per step over k=2) for +33% verified tokens and +27 MiB/seq of state.
- Draft TP, draft quantization and `probabilistic` draft sampling are all irrelevant here: 1 GPU, built-in MTP head, greedy client.

### Gaps
- No upstream acceptance numbers specifically for Qwen3.5-9B MTP in thinking mode at temp 0. Our own measurements are the best data.
- No source measured how acceptance varies between the thinking span and the answer span.

---

## Q5. Spec-decode CUDA-graph and scheduler overheads at large batch on A100

### Takeaway
CUDA-graph coverage is not a problem in 0.32. With the default (unset) capture size, the V2 runner appends FULL decode graphs on a request-count grid up to `max_num_seqs`, scaled by the query length. E020 (k=3) captured 35 decode graphs, which matches that grid exactly for 256 requests. So no capture-size change is needed, and an explicit `--max-cudagraph-capture-size` should NOT be set, because it suppresses the appended decode sizes.

Several open PRs speed up the Qwen3.5 MTP step (fused kernels instead of torch.compile: −17% to −38% MTP decode step on 9B) or remove host syncs. None is in 0.32 (qwen3_5.py still uses `@support_torch_compile`).

### Cited Findings
- **Capture-size default:**
  - The token grid uses `default_max_graph_size = 1024 if device_capability_family(100) else 512`, with `max_cudagraph_capture_size = min(max_num_seqs*decode_query_len*2, default_max_graph_size)`.
  - For spec decode, uniform-decode sizes are appended separately from a request-count grid: `{1,2,4} + range(8, min(max_reqs+1,256), 8) + range(256, max_reqs+1, 16) + [max_reqs]`, scaled by the query length.
  - Code comment: these decode sizes are "Populated only when the default is computed here, so an explicit capture range is left exactly as configured".
  — LOCAL `config/vllm.py` L2346-2395.
- **Our runs:**
  - Champion/E040 log `max_cudagraph_capture_size: 512` and `cudagraph_mode FULL_AND_PIECEWISE`.
  - E020 (k=3) captured "decode CUDA graphs (FULL): 35/35". With `max_num_seqs=256` the request grid is 3 + 31 + 1 = 35 sizes, so FULL decode graphs cover up to 256 requests (1024 tokens) at k=3.
  - E020 also captured prefill FULL 49/49 and PIECEWISE 51/51 (the 512-token grid).
  - Attention backend is FLASH_ATTN.
  — REPO `experiments/E040/server.log`, `experiments/E020/server.log`.
- **#51012 (open):** "Deprecate torch.compile for Qwen3.5 models". Replaces it with explicit fused kernels and breakable CUDA graphs. Reports decode ITL −3.0% to −8.8%, prefill TTFT −7.0% to −8.9%, and **MTP decode step −17% to −38% for the 9B** (GB200), with token-identical greedy output. — [PR #51012](https://github.com/vllm-project/vllm/pull/51012) (via GitHub search API summary). Local `model_executor/models/qwen3_5.py` L213 still has `@support_torch_compile`, so it is not in 0.32 — LOCAL.
- **#57352 (open):** the FlashInfer builder copies `seq_lens` from GPU each step under spec decode, forcing a sync. The fix gives +1% to +20% output tok/s with MTP-3/EAGLE-3 on RTX GPUs. Not run on Hopper/Blackwell. Applies to FlashInfer; we use FLASH_ATTN. — [PR #57352](https://github.com/vllm-project/vllm/pull/57352)
- **#35387 (open issue):** MTP caused a 76% latency regression on Qwen3-Next-80B FP8 (0.894 s → 1.578 s). The reporter suspects a `num_accepted_tokens` CPU copy before `mamba_postprocess`. — [Issue #35387](https://github.com/vllm-project/vllm/issues/35387) (via GitHub search API summary)
- **#58863 RecoverSSM:** removes about 0.27 ms of a 0.8 ms per-step state commit at c=8 (H200), because the per-draft-position state write-back is dropped. — [PR #61309](https://github.com/vllm-project/vllm/pull/61309)
- **#60566 (open) "Screened lm_head":** FP8 screen plus exact BF16 rows. +3.9% with MTP k=3 on DGX Spark, neutral/slightly negative on GB200/GB300, +0.64 GB memory. — [PR #60566](https://github.com/vllm-project/vllm/pull/60566) (via GitHub search API summary)
- **General overheads:** vLLM overhead incl. scheduling is 3-12% of SD execution, shrinking with batch. Verification dominates (42-95%). — [arXiv 2601.11580v2](https://arxiv.org/html/2601.11580v2)

### Inferences
- **Leave the capture size at its default for both k=2 and k=3.** Setting `--max-cudagraph-capture-size` explicitly would drop the appended decode sizes. At k=2, most token-grid entries are not multiples of 3, so it could lose FULL decode coverage.
- **Mixed steps run without graphs.** At peak, mixed steps (a ≤1024-token prefill chunk plus about 130 x 3 decode tokens) exceed the 512-token piecewise grid. That is probably minor at that size, since such steps are compute-bound.
- **On A100 at our peak, GDN's per-draft state write-back is a real per-step cost.** Writing (K+1) x 24 x ~1 MiB per sequence each step is about 10 GB/step at k=2 and 130 seqs, if fully written. RecoverSSM targets exactly this cost.
  - This bandwidth estimate is an inference and should be confirmed with nsys.
- **#51012 would be the biggest pure-kernel MTP win for the 9B**, but it is unmerged, benchmarked on GB200, and not in 0.32. Backporting it is a large change.

### Gaps
- No A100-specific measurements found for spec decode overheads, dynamic SD, or Qwen3.5 MTP step time at high batch.
- No measurement of the GDN state write-back's share of step time on our server.
