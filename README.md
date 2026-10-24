# GreenMind LLM Inference Optimization Hackathon (Team 18): MANGO1.5-Qwen3.5-9B on 1× A100-SXM4-40GB

**Final submission: OFFICIAL Final S = 99.38**, attempt `20261024T164844-c109c6` (2026-10-24 16:48 UTC).
- **Score terms:** A 1.00, s_E 1.00, s_T 0.969, s_L 1.00.
- **Energy and SLOs:** 157.5 J/request (official baseline 467). 294/300 requests on time, with p95 TTFT 1.21 s and p95 TPOT 37 ms.
- **Errors:** none. 0 truncated and 0 errors.

The launch script is [`scripts/serve_final.sh`](scripts/serve_final.sh). It is identical to [`configs/E046.sh`](configs/E046.sh) and `configs/champion.sh`. The power limit is the GPU default (400 W), with no clock lock.

## Approach

We made seven changes on top of a default `vllm serve`. Each one was measured on our local grader, and the key steps were confirmed on the official grader.

| # | Change | Why it helps | Evidence |
|---|---|---|---|
| 1 | **MTP self-speculative decoding, 2 draft tokens** (the model's own BF16 MTP head) | About 2.6 tokens per step at an acceptance of 0.87/0.74, giving about −30% J/request and about −25% TPOT. k=3 needs one more recurrent-state slot per sequence and collapses under final-track load. k=1 is slower. | E005–E009, E020, E032 |
| 2 | **`--mamba-ssm-cache-dtype bfloat16`** | With MTP, each sequence holds k+1 Gated-DeltaNet recurrent states. In fp32 that capped concurrency at about 46 sequences (KV 98%, queueing). bf16 halves it: peak KV drops from 98% to 56% and TTFT fails from 59 to 33. | E006→E007 |
| 3 | **Prefill scheduling: `--max-num-batched-tokens 4096 --long-prefill-token-threshold 1024`** | Short prompts were queuing behind the prefill of a single 18–20K-token prompt (a "TTFT staircase"). Capping each request's chunk lets new arrivals start in the next step. Local val goes from 63 to 80 of 100 on time (E008 → E017). Officially, E008 scored Val S 31.5 (65/100) and E017 scored Final S 59.0 (215/300). | E010, E011, E016–E019, E024, E026, E028 |
| 4 | **No prefix caching, `--language-model-only`, `--max-model-len 41984`** | The prefix-cache hit rate was 3% on this workload, and align mode costs memory. Skipping the vision tower frees memory. 41984 tokens = 32K prompt + 8192 output. | E006 |
| 5 | **Server-side thinking budget of 3072 tokens** ([`serving/vllm_think_budget.py`](serving/vllm_think_budget.py)) | Requests that thought all the way to the 8192-token cutoff always failed. They also stretched the energy window by about 25 s at about 300 W. vLLM's built-in budget forces `</think>` instead. This took the official Final from **59.0 to 88.0**. 1536–2048 is too tight (answers then run on to the cutoff), 5120 is worse. | E027, E030, E031, E038, E039 |
| 6 | **Team-made INT8 W8A8 weights** ([`quantization/`](quantization/)) | A100 has native INT8 tensor cores, and vLLM uses `CutlassInt8ScaledMMLinearKernel` for them (about 2× BF16). This removed the last bottleneck, compute-bound prefill at peak. The official Final went from **88.0 to 99.38**. Weight-only quantization (organizer AWQ W4A16, online FP8) runs on Marlin kernels instead, which are slower at batch and also cost accuracy. | E045, E046 (vs E001/E004/E037) |
| 7 | Served model name `mango` | Label only. It has no effect on serving or measurement. | E046 |

**Trade-offs and caveats:**
- **Accuracy:** the W8A8 weights keep overall accuracy at A = 1.00 (official 88.3% vs baseline 87.2%). Official long-context retention is 0.71, though: 10 of 15 correct, vs 0.93 locally. That class has fewer than 20 questions, so it isn't gated, but it would lower A on a task set with 20 or more long-context items. Locally, RAG and reasoning are also about 0.04 below BF16 (see E045/E046).
- **The thinking budget changes generations.** Requests whose reasoning reaches 3072 tokens are forced to answer. In local E017 runs no request finished naturally between 5K and 8192 tokens, and the 3–7 that hit the 8192 cutoff always failed. TODO: count how many requests hit the 3072 budget in the official run. The team lead authorized it on 2026-10-24. **TODO (team):** record the organizers' ruling, since the briefing says the grader fixes thinking mode.
- **Power:** the run used the default 400 W limit with no clock lock. Power caps (250/300/350 W) and an SM clock lock (1215 MHz) saved energy but lost more on-time requests than they gained. Officially, the clock lock scored 52.5 vs 59.0 for the same config without it (E023).

## Results vs baseline

**Official results (organizer grader).** Validation and Final are labelled. All Final runs have 300 requests.

| Config | Track | S | A | s_E | s_T | s_L | η vs base | J/req | SLO-met | p95 TTFT | p95 TPOT | Accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Default `vllm serve` BF16 (E000) | Val | (baseline) | 1.00 | – | 0.51 | 0.59 | 0.89× | 467 | 64/100 | 2.06 s | 48 ms | 87.2% |
| E008: MTP2 + bf16 SSM | Val | 31.5 | 1.00 | 0.22 | 0.51 | 0.59 | 1.27× | 333 | 65/100 | 2.02 s | 38 ms | 88.9% |
| E017: + prefill scheduling | Final | 59.0 | 1.00 | 0.53 | 0.62 | 0.80 | 1.78× | 247 | 215/300 | 1.91 s | 59 ms | 88.8% |
| E023: E017 + SM clock lock 1215 MHz | Final | 52.5 | 1.00 | 0.54 | 0.37 | 0.69 | 1.80× | 186 | 164/300 | 2.15 s | 74 ms | 89.1% |
| E026: E017 with prefill cap 768 | Final | 46.6 | 1.00 | 0.37 | 0.55 | 0.81 | 1.50× | 264 | 193/300 | 2.22 s | 63 ms | 90.3% |
| E030: E017 + thinking budget 3072 | Final | 88.0 | 1.00 | 0.96 | 0.71 | 0.85 | 2.87× | 173 | 242/300 | 1.85 s | 57 ms | 90.7% |
| E033: E030 + gpu util 0.96 | Final | 81.6 | 1.00 | 0.90 | 0.66 | 0.76 | 2.68× | 176 | 231/300 | 1.90 s | 60 ms | 90.4% |
| E039: E030 with budget 1536 + "answer now" phrase | Final | 86.0 | 1.00 | 0.97 | 0.64 | 0.80 | 2.92× | 159 | 226/300 | 1.90 s | 60 ms | 88.7% |
| **E046: E030 + team W8A8 INT8 weights (FINAL)** | **Final** | **99.38** | 1.00 | 1.00 | 0.97 | 1.00 | 3.82× | **157.5** | **294/300** | **1.21 s** | **37 ms** | 88.3% |

The official per-class accuracy and retention for E046 are in [`results/official/E046_final_20261024T164844-c109c6.json`](results/official/E046_final_20261024T164844-c109c6.json):

| Class | Accuracy | Retention |
|---|---|---|
| chat | 0.86 | 0.96 |
| rag | 0.95 | 1.04 |
| longctx | 0.67 | 0.71 |
| reasoning | 0.91 | 1.03 |
| agentic | 0.85 | 1.06 |
| thai | 0.97 | 1.16 |

Official energy was 47.2 kJ over 144.7 s at an average of 326.5 W, with a peak of 432 W and the power limit at 400 W. The registry of all official results is in [`results/official/official_list.txt`](results/official/official_list.txt).

**Local grader, key runs.** Local S is relative to our own local baseline and is not comparable to official S. Throughput is generated tokens divided by the energy window.

| Run | Track | SLO-met | J/req | Energy | Gen. tokens/s | Tokens/J | p95 TTFT | p95 TPOT | Accuracy |
|---|---|---|---|---|---|---|---|---|---|
| Baseline `20261024-120300-vllm-val-baseline` | Val | 78/100 | 463 | 46.3 kJ | 755 | 2.40 | 1.30 s | 50 ms | 81% |
| Baseline `20261024-124012-baseline_bf16_default` | Final | 106/300 | 278 | 83.5 kJ | 1290 | 3.85 | 2.87 s | 131 ms | 81.3% |
| E030 `20261024-192201-e030` | Val | 83/100 | 258 | 25.8 kJ | 1409 | 3.98 | 1.49 s | 50 ms | 80% |
| E030 `20261024-192326-e030` | Final | 123/300 | 195 | 58.6 kJ | 1730 | 4.68 | 4.23 s | 81 ms | 81.0% |
| E045 `20261024-232535-e045` | Val | **100/100** | 205 | 20.5 kJ | 1559 | 4.59 | 1.11 s | 36 ms | 81% |
| E046 `20261024-234133-e046-final` | Final | **257/300** | 179 | 53.8 kJ | 1751 | 5.09 | 1.63 s | 50 ms | 82.3% |

Every experiment has its card, result, launch command and server log under `experiments/<id>/`. The summary table is [`EXPERIMENTS.md`](EXPERIMENTS.md), and the lessons are in [`KEY_FINDINGS.md`](KEY_FINDINGS.md).

## Model weights (disclosure)

| Weights | Provided by | Link | Used in final? |
|---|---|---|---|
| `CMKL/MANGO1.5-Qwen3.5-9B` (BF16, rev `c0e66f0d`), Apache-2.0 | Organizers (original) | https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B | Source of our quantization. Served in E005–E044. |
| `CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp` (rev `c44c8a80`), Apache-2.0 | Organizers (provided quantization) | https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B-AWQ-W4A16-mtp | No. Rejected in E001/E004 (agentic accuracy, slower). |
| **`suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8`** (rev `1909e853`), Apache-2.0 | **Made by our team during the event** (uploaded 2026-10-24 16:14 UTC) | https://huggingface.co/suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8 | **Yes (final).** |

- **How the W8A8 weights were made:** GPTQ W8A8 with llm-compressor, using INT8 per-channel weights and dynamic per-token INT8 activations on the attention and MLP linear layers. `lm_head`, embeddings, the vision tower, the Gated-DeltaNet `linear_attn` projections and the MTP head stay BF16. Recipe and reproduction steps are in [`quantization/`](quantization/).
- **Speculative draft:** no external draft model. The draft is the model's own MTP head, which is part of the official checkpoint and is kept in BF16 inside the W8A8 repo as `model_mtp.safetensors`.
- **Calibration data:** open datasets only, with no test items. 512 samples of up to 2048 tokens, split 45% Thai / 35% English chat / 20% STEM:

| Dataset | License (from Hub metadata) | Share |
|---|---|---|
| [airesearch/WangchanThaiInstruct](https://huggingface.co/datasets/airesearch/WangchanThaiInstruct) | CC-BY-SA-4.0 | 45% |
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | 35% |
| [garage-bAInd/Open-Platypus](https://huggingface.co/datasets/garage-bAInd/Open-Platypus) | **TODO:** none declared in Hub metadata; the card lists per-subset licenses | 20% |

- **TODO (team):** the W8A8 repo is currently **private**. Make it public, or grant the organizers access, before judging.

## Setup and launch

The hardware and software is listed in full in [`requirements/system.md`](requirements/system.md), with an exact `pip freeze` in [`requirements/vllm-env.lock.txt`](requirements/vllm-env.lock.txt):

| Item | Version |
|---|---|
| GPU | 1× A100-SXM4-40GB |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| OS | Ubuntu 24.04 |
| Python | 3.12.3 |
| **vLLM** | **0.32.0** |
| torch | 2.13.0+cu130 |
| transformers | 5.19.0 |
| flashinfer | 0.7.0.post1 |
| compressed-tensors | 0.17.0 |

```bash
# 1. Python env (the organizer VM image ships ~/vllm-env with vllm 0.32.0; to recreate it:)
python3.12 -m venv ~/vllm-env && ~/vllm-env/bin/pip install -r requirements/vllm-env.lock.txt   # TODO: not re-tested from scratch
#    FlashInfer JITs its sampler at first use: needs ninja (in the venv) and nvcc (/usr/local/cuda/bin) -> handled by the script.

# 2. Weights (private repo for now: needs a read token once; nothing is stored in this repo)
HF_TOKEN=<read-token> ~/vllm-env/bin/hf download suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8 --revision 1909e853b46b2618457ba4205e48a61df6a0d888

# 3. GPU power: default limit, no clock lock (held constant for the whole run)
scripts/power.sh reset          # -> 400 W on A100-SXM4-40GB, application clocks unlocked

# 4. Launch the final server (port 8000, model name "mango"); ~1 min warm start, ~3-5 min on a cold compile cache
scripts/serve_final.sh
#    or, with stop-old-server + wait-for-health:  scripts/restart.sh final
```

Once the server is up, `local_grader/grader.py health --url http://localhost:8000` checks streaming with usage, tool calling and NVML energy, and every check passed on our VM.

**Verified on 2026-10-25** ([`experiments/final/README.md`](experiments/final/README.md)): `scripts/power.sh reset && scripts/restart.sh final` started a server whose arguments are identical to the official E046 process. Health passed. A local val run (`20261025-002935-final-verify`) scored 100/100 on time, η 2.19×, 210 J/request, p95 TTFT 1.05 s and p95 TPOT 33 ms.

**What the endpoint supports (verified with the health check and both graders):**
- **Chat API:** OpenAI-compatible `POST /v1/chat/completions` and `GET /v1/models`.
- **Streaming with usage:** `"stream": true` with `stream_options.include_usage`, which reports prompt, completion and reasoning tokens.
- **Tool calling:** `tool_choice: "auto"` via `--enable-auto-tool-choice --tool-call-parser qwen3_coder`.
- **Reasoning split:** reasoning arrives separately from content via `--reasoning-parser qwen3`.

The grader's `max_tokens`, temperature and thinking mode are left untouched. The only server-side change is the default `thinking_token_budget` of 3072, applied when a request doesn't set one.

## Repository layout

| Path | Contents |
|---|---|
| `scripts/serve_final.sh` | **Final launch command** (env-overridable paths; defaults = our VM) |
| `serving/vllm_think_budget.py` | `vllm serve` wrapper. It sets a default `thinking_token_budget` (env `THINK_BUDGET`) and has an optional, env-gated, off-by-default short-first scheduler patch that was rejected in E040. |
| `configs/` | Exact launch command of every experiment (`E001.sh` … `E046.sh`, `champion.sh` = `E046.sh`, `final.sh` → `scripts/serve_final.sh`, `trtllm/`). |
| `scripts/` | `restart.sh` (stop the server, wait for an idle GPU, start a config, wait for `/health`), `run.sh` and `exp.sh` (local grader runs), `power.sh`, `warmup.py` (E002, not used) |
| `quantization/` | Team W8A8 quantization toolkit (vendored from github.com/suppakorn-tae/mango-quant @ `6ffc9c9`) and the artifact's recorded recipe |
| `experiments/` | `log.jsonl` (append-only event log), `render.py`, and per experiment `card.md`, `result.md`, `config.sh`, `server*.log` and grader output |
| `results/official/` | Official grader exports (full JSON for E039 and E046) and the official results registry |
| `results/local/` | Local grader exports: `history_{val,final}.txt` and `runs/<run_id>/summary.json` + `report.html` for all 116 local runs, plus raw `records.jsonl`/`power.csv` for the key runs |
| `local_grader/` | Snapshot of our local grader (`aef5a17`) with its frozen traces, used for every local number |
| `docs/research/` | Background research notes by a teammate (literature and vLLM-source reading). These are not results. |
| `CHAMPION.md`, `HANDOFF.md`, `EXPERIMENTS.md`, `KEY_FINDINGS.md` | Current champion, handoff notes, experiment table, lessons |

## What didn't work, and why

All of these are logged with numbers in `experiments/<id>/result.md`.

- **Organizer AWQ-W4A16 checkpoint (E001, E004):** agentic accuracy fell to a deterministic 2/20 (BF16 scores 3–7/20). Marlin W4A16 is also slower than BF16 at serving batch sizes on A100.
- **Online FP8 weights (E037):** A100 has no FP8 compute, so this runs as Marlin W8A16. p95 TPOT was 65 ms on val and 131 ms on the final, and reason/long accuracy dropped. **Lesson:** use activation-quantized INT8 (W8A8) on Ampere, not weight-only quantization.
- **Pre-measurement warmup (E002):** the apparent "warm" gains were the prefix cache replaying the identical trace, not JIT warmup.
- **Smaller or larger prefill budgets** (2048/3072/5120, E003/E019/E024/E028/E035): too small makes requests queue, too large stalls decoders. Under final-track load, both 3072 and 5120 collapse into KV-full backlogs.
- **Prefix caching with mamba align mode (E006):** 3% hit rate on this workload.
- **MTP k=1 (E009)** is slower. **MTP k=3 (E020, E032)** improves TPOT but collapses on memory at final peak. **Dynamic k by batch size (E043/E044)** shrinks the KV pool from 475K to 368K tokens and collapses as well.
- **Power caps of 250/300/350 W and an SM clock lock at 1215 MHz (E012–E014, E022, E023):** prefill is compute-bound, so energy savings come with an equal loss of on-time requests. The clock lock scored 52.5 officially vs 59.0.
- **Admission cap `--max-num-seqs 112` (E036):** fewer TPOT fails but more TTFT fails.
- **Raising `--gpu-memory-utilization`:** 0.95/0.96 brought no official gain (81.6 vs 88.0, within official noise), and 0.98 ran out of memory under load (E015, E025, E033, E034).
- **fp8 KV cache (E021):** on Ampere it forces FlashInfer attention and downgrades CUDA graphs under spec decode.
- **Thinking budgets of 1536/2048 (E031, E038):** forced-early answers ran away to the 8192 cutoff. The "answer now" phrase in E039 fixed that, but the official score was 86.0 vs 88.0.
- **Short-first scheduler patch (E040)** and **cap 768 (E026)** just move TTFT misses onto long prompts.
- **SGLang 0.5.22 (E041):** best val 58/100 vs vLLM 80/100. GDN spec-decode states take about 96 MB per slot, so only 80 slots fit. The MTP draft loads its own 4.2 GB of weights. It has no per-request prefill cap.
- **TensorRT-LLM 1.4.0rc1 (E042):** runtime JIT stalls every fresh server for a whole trace (24/100 cold, 56/100 warm), and agentic accuracy was 2/20. It also depends on closed NVIDIA binary wheels.
- **Calibration lesson:** local edges of 12–16% did not carry over officially (E026: 46.6 vs 59.0; E033: 81.6 vs 88.0), which suggests official run-to-run noise of roughly ±10%. Only structural changes carried over: the thinking budget (+49%) and W8A8 (+13%).
