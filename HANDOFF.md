# Handoff (updated 2026-10-25 ~00:30 +07)

## State
- **Final submission: E046**, OFFICIAL Final S **99.38** (attempt `20261024T164844-c109c6`).
  - Weights: team W8A8 INT8 weights `suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8` rev `1909e853`.
  - Serving: vLLM 0.32.0, MTP k=2, bf16 SSM state, prefill 4096/1024, thinking budget 3072, served as `mango`.
  - Power: default 400 W, no clock lock.
- **Launch:** `scripts/power.sh reset && scripts/restart.sh final`. `restart.sh` stops whatever owns :8000, waits for an idle GPU, starts the config detached, and waits for `/health`. `final` is `configs/final.sh`, which points to `scripts/serve_final.sh`.
  - **Gotcha:** don't `pkill -f "vllm serve"` from a shell whose own command line contains that string. It kills the shell.
- **Fallbacks:**

| Config | Weights | Official Final S |
|---|---|---|
| `configs/E030.sh` | original BF16, same flags | 88.0 |
| `configs/E017.sh` | original BF16, no thinking budget | 59.0 |

## Official results, all of them

| ID | Track | S | Note |
|---|---|---|---|
| baseline | val | – | 64/100, 467 J/req |
| E008 | val | 31.5 | |
| E017 | final | 59.0 | |
| E023 | final | 52.5 | clock lock |
| E026 | final | 46.6 | cap 768 |
| E030 | final | 88.0 | |
| E033 | final | 81.6 | util 0.96 |
| E039 | final | 86.0 | budget 1536 |
| **E046** | **final** | **99.38** | |

Details are in `results/official/` and in the README table.

## Open items / TODO (team)
1. **Make `suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8` public**, or share it with the organizers. The deliverables require HF links to the optimized weights.
2. **Organizer ruling on the server-side thinking budget.** The briefing says the grader fixes thinking mode. It was authorized by the team lead and should be confirmed with the organizers.
3. **Revoke the HF read token** that was pasted into the agent chat on 2026-10-24. It isn't stored anywhere in this repo.
4. **Quantization provenance:**
   - The exact llmcompressor / torch / transformers versions and command used for rev `1909e853`. The vendored toolkit commit postdates the artifact by 54 minutes.
   - The Open-Platypus license.
   - Any A/B Thai-benchmark results from `quantization/eval_ab.sh`.
5. **Official evidence:** attempt IDs and exports or screenshots for the official runs other than E039 and E046, plus the official Final baseline numbers if they're available.
6. **Recreating the venv** from `requirements/vllm-env.lock.txt` on a fresh machine is untested. The VM image shipped the env.

## Untested ideas (ranked)
1. **Long-context retention.** It's officially 0.71, the only weak class. Things to try:
   - Re-quantize with long RAG/document samples in the calibration set (`--calib-jsonl` with open long-document data, not test items).
   - Keep the full-attention layers' `o_proj`/`qkv` in BF16.
   - Check locally whether the drop comes from W8A8 or the thinking budget.
2. **With W8A8 there is memory headroom.** The KV pool is 503–609K tokens and the final peak had 0 waiting. MTP k=3 and dynamic k were memory-limited on BF16 and could be re-tested on W8A8.
3. Repeat the official run of E046 to measure noise. The best finished run counts.

## Pitch notes
- **Diagnose first.** We read the vLLM scheduler log at peak (KV %, waiting queue, prompt vs generation throughput), per-request TTFT staircases and the per-phase energy split, and changed one variable at a time. That's 47 experiments, all logged, including failures.
- **Three structural wins, in order:**
  1. bf16 recurrent-state cache, so MTP fits in memory on a GDN hybrid model.
  2. Thinking budget: cut-off requests always fail and burn the energy tail. 59 → 88.
  3. INT8 W8A8: the right quantization for Ampere tensor cores. 88 → 99.4.
- **Energy:** 157.5 J/request official vs the 467 baseline (−66%). 294/300 requests on time. Accuracy 88.3% vs 87.2%.
- **Engine bake-off:** vLLM 0.32 (80/100 val) beat SGLang 0.5.22 (58/100) and TensorRT-LLM 1.4.0rc1 (56/100 warm) on this model and GPU.
