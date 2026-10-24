# Raise peak compute, not scheduling, to approach 95

Getting from 88 to 95 means raising latency attainment at peak, sT·sL, from about **0.60 to about 0.875 (+45%)**. If sE also reaches its cap of 1.0, the bar drops to about **0.77 (+28%)**. No scheduling or flag tweak in the notes is credibly worth that much. The only untried lever large enough is **INT8 W8A8 weights made by us from the MANGO checkpoint**. Peak steps are compute-bound, and INT8 is the only format that runs GEMMs on A100's INT8 tensor cores (about 2x the BF16 rate). It speeds up prefill and decode verification together, and it lowers energy per token. Our roofline estimate of 1.15-1.35x faster peak steps comes from the team's calculation, not a measurement. It puts S at roughly **91-96, with a central estimate of 93-94**, so 95 is reachable only if INT8 lands in the upper half of that range and accuracy holds. Before INT8, fix a measurement problem that our own logs expose. **All ten BF16 launches at utilization 0.92 that compiled torch.compile graphs cold (E005-E007, E009, E010, E019-E021, E024, E028) got a KV pool about 3.8 GiB smaller** (13.3-13.6 vs 16.9-17.4 GiB). Every launch that loaded graphs from cache got the normal pool. Several "rejected" configs were therefore judged on a crippled server, and the first launch of INT8 or any new flag set will be too. So every measured launch must follow a throwaway warm-up launch. After INT8, two cheap knobs aimed only at peak are worth one local cycle each: a batch-size-dependent MTP depth (k=2 normally, k=1 at peak) and a decodes-first reordering of the running queue. Each is worth low single-digit percent at best, and only if step logs confirm the mechanism. Be honest about the starting point too: E030 has no clean official replicate, and the ±10% noise estimate comes from runs with changed configs. 88.0 may be a favourable draw, and 95 is a stretch target rather than an expected outcome.

## 95 demands near-perfect peak attainment, and TTFT is the binding term

The score decomposes cleanly. With A = 1.00 and sE = 0.96, sE^0.6 = 0.976. So E030's 88.0 implies (sT·sL)^0.2 ≈ 0.902, i.e. **sT·sL ≈ 0.60**, which matches the reported 0.71 × 0.85 ([CHAMPION.md](file:///home/researcher/repo/CHAMPION.md)). Reaching 95 needs (sT·sL)^0.2 ≈ 0.974, i.e. **sT·sL ≈ 0.875**. If sE reached 1.0 it would need 0.774. The 0.2 exponent is unforgiving: a 10% gain in sT moves S by only about 2% ([KEY_FINDINGS.md](file:///home/researcher/repo/KEY_FINDINGS.md)).

| Target S | sE assumed | sT·sL required | Example split |
|---|---|---|---|
| 88.0 (today) | 0.96 | 0.60 | sT 0.71, sL 0.85 |
| 91 | 0.96 | 0.71 | sT 0.80, sL 0.89 |
| 95 | 0.96 | 0.875 | sT 0.92, sL 0.95 |
| 95 | 1.00 | 0.774 | sT 0.84, sL 0.92 |

Official runs give a rough mapping from on-time count to sT. E017 → E030 (215 → 242 on time) moved sT from 0.62 to 0.71, and E039 → E030 (226 → 242) moved it from 0.636 to 0.71 ([EXPERIMENTS.md](file:///home/researcher/repo/EXPERIMENTS.md)). That is **about 0.003-0.005 sT per extra on-time request**. On that slope, sT 0.92 means roughly 285-300 of 300 requests on time. Even with sE at its cap, sT 0.84 means roughly 270-280 on time, i.e. **30-40 more peak requests rescued**.

sL is the geometric mean of a TTFT term and a TPOT term. The official E039 detail shows s_l_ttft 0.77 and s_l_tpot 0.83, whose geometric mean is the reported sL of 0.80 ([KEY_FINDINGS.md](file:///home/researcher/repo/KEY_FINDINGS.md)). **TTFT is the weaker term**, with p95 TTFT at 1.30x its SLO. E040 diagnosed it: throttling in-flight long prefills so short prompts go first only moved TTFT misses onto the long/RAG prompts, which leads to the conclusion that **peak prefill capacity is the hard limit**. About 65K long-prompt tokens arrive within 3 s at peak, and the misses are marginal (TTFT 0.52-0.88 s against 0.51-0.65 s SLOs; TPOT 51-72 ms) ([KEY_FINDINGS.md](file:///home/researcher/repo/KEY_FINDINGS.md)). Marginal misses are good news, because a uniform per-step speedup converts them directly. The constraint is just as clear, though: reordering a fixed amount of prefill work only redistributes misses. The levers that move this term must **add tokens per second at peak**.

The best evidence for how strongly peak attainment responds to compute is official, and it runs in the wrong direction. E023 locked the SM clock at 1215 MHz, about 14% below the roughly 1.41 GHz the GPU otherwise holds at peak. That moved official sT from 0.62 to 0.37 and sL from 0.80 to 0.69, so **sT·sL fell from about 0.50 to 0.26 (-49%) for a ~14% compute cut** ([EXPERIMENTS.md](file:///home/researcher/repo/EXPERIMENTS.md); [GreenLLM](https://arxiv.org/html/2508.16449v1) for the A100's ~1.41 GHz maximum). The response will flatten as attainment approaches 1, so gains on the way up will be smaller than that loss. Still, it is measured proof that peak SLO attainment is steeply elastic to raw throughput, and that elasticity is the case for buying compute.

The peak regime explains why. At about 130-140 sequences with MTP k=2, every weight GEMM sees M ≈ 390-420 verify tokens, plus up to 4096 chunked-prefill tokens. That is well past A100's BF16 ridge of about 156-200 FLOP/byte, so **peak steps are compute-bound**. A rough tally of 105 TFLOP/s of prefill plus 30 TFLOP/s of decode puts the server at about 43% of BF16 peak. That figure is the team's inference, recorded in the kernels note. It also accounts for three past results: clock and power caps lost SLO about 1:1 with energy (E012/E014/E022/E023), Marlin W4A16/W8A16 lost (E001/E037), and published "decode is frequency-insensitive" results did not transfer. The A100 study GreenLLM is consistent with this: its decode-energy savings shrink from 20-25% at low TPS to 8-12% at high TPS ([GreenLLM](https://arxiv.org/html/2508.16449v1)).

## Cold compile caches cost every new config 3.8 GiB of KV pool

The "Available KV cache memory" line in our server logs is **bimodal**, and the cause is visible in the same logs. BF16 launches that printed "Compiling a graph for compile range ..." profiled **13.3-13.6 GiB** (E005-E007, E009, E010, E019, E020, E021, E024, E028), and E036 profiled 15.39 GiB. Every BF16 launch that loaded compiled graphs from cache profiled **16.9-17.4 GiB** at utilization 0.92 ([experiments/](file:///home/researcher/repo/experiments/)). In the cold launches, vLLM's own memory accounting shows **weights + non-torch at about 19.8 GiB and peak activation at about 3.0 GiB**, against 17.6 and 1.3-1.4 GiB for warm ones. Free device memory at startup was identical (39.08/39.49 GiB) in both groups, so the gap comes from the cold compile happening inside the launch, not from a leftover process. The cleanest pair is E020 vs E032: identical vLLM flags (k=3), the first launch compiled and got 13.55 GiB, and the second hit the cache and got 17.43 GiB. The mechanism is our inference: compile-time allocations are counted during memory profiling. The correlation, though, has no exceptions among the BF16 runs.

The bias falls systematically on **the first launch of any flag set that changes the compiled graph** (MAX_BT, MTP depth, max-num-seqs, KV dtype). Prefill-cap changes like E026 reuse the cache. Several verdicts were reached on such launches, citing **KV-full queueing or backlog collapse**: E024 ("KV-full queueing → TTFT 11.7 s"), E028 ("final backlog collapse (36 waiting)"), E019 ("re-creates queueing") and E020 ("KV 98.7%") ([EXPERIMENTS.md](file:///home/researcher/repo/EXPERIMENTS.md)). Some still stand. E032 re-ran k=3 warm at 17.43 GiB and still collapsed. E035 re-ran MAX_BT 3072 warm at 19.06 GiB and still lost. E021 failed on a CUDA-graph downgrade rather than memory. Others were never re-tested warm. E028 (MAX_BT 5120) was not. E036 (`--max-num-seqs 112`) also compiled cold, at 15.39 GiB. **E009's static k=1 result was measured cold**, so the "+23% J/req" figure that argues against k=1 is not clean. On the small pool, k=2 holds roughly 25-30 fewer concurrent sequences (3.8 GiB at about 81 MiB of GDN state each, our estimate). Every official run so far profiled 17.36 GiB or more, so this has distorted local A/B tests, not official scores ([champion/server_2045.log](file:///home/researcher/repo/experiments/champion/server_2045.log)).

The fix is procedure first and a flag second. **Before every measured launch of a new config, including INT8, do a throwaway launch that reaches `/health` and stop it, so the compile cache is warm.** Never clear `~/.cache/vllm` before an official run. Then, as a guard, pin the pool with `--kv-cache-memory-bytes`, which sets it directly and "ignores gpu_memory_utilization" ([vllm/config/cache.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/config/cache.py)). The server's own hint prints `--kv-cache-memory=...`, but the argparse flag is `--kv-cache-memory-bytes` ([vllm/engine/arg_utils.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/engine/arg_utils.py)). Pin to the champion's 17.36 GiB (18640000000 bytes), not higher: E033 (E030 plus utilization 0.96, an 18.94 GiB pool) scored 81.6 officially, and E034 at 19.73 GiB OOMed under load. On a warm start the budget is 17.59 GiB weights/non-torch + 0.66 GiB graphs + 1.4 GiB activation + 17.36 GiB pool ≈ 37 of 39.5 GiB. A cold start would need about 40.8 GiB and cannot fit ([champion/server_2045.log](file:///home/researcher/repo/experiments/champion/server_2045.log)). Pinning therefore turns a silently crippled launch into a loud failure, which is the right trade, provided the warm-up step comes first. None of this raises the champion's mean score. It makes every later comparison trustworthy. Among the cold-run rejections, only E028's MAX_BT 5120 is worth a warm re-test, and as a one-flag edge it must still clear the 20% bar below before it earns an official run.

Instrumentation belongs in the same phase. The SLO note's code reading of the 0.32 scheduler finds that `schedule()` walks `self.running` in admission order and stops when the 4096 budget is gone. Four long partial prefills at the 1024 cap can therefore consume a whole step and leave **later-admitted decodes with zero tokens** ([vllm/v1/core/sched/scheduler.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py)). Nobody has measured whether this happens. A per-step log at peak settles it, and it also yields the step-time model that any SLO-aware policy needs. The existing wrapper already patches `Scheduler.schedule` and runs inside the spawned EngineCore, the E040 pattern ([tools/gray/vllm_think_budget.py](file:///home/researcher/repo/tools/gray/vllm_think_budget.py)). Logging can reuse that hook:

```python
# add to tools/gray/vllm_think_budget.py, gated by env STEP_LOG=/path/steps.jsonl
import json, time
from vllm.v1.core.sched.scheduler import Scheduler
_orig_sched = Scheduler.schedule
_f = open(os.environ["STEP_LOG"], "a") if os.environ.get("STEP_LOG") else None
def _logged_schedule(self, *a, **k):
    out = _orig_sched(self, *a, **k)
    if _f:
        sched = out.num_scheduled_tokens            # dict req_id -> tokens this step
        dec = [r for r in self.running if not r.is_prefill_chunk]
        starved = sum(1 for r in dec if r.request_id not in sched)
        _f.write(json.dumps({"t": time.monotonic(), "tok": out.total_num_scheduled_tokens,
            "running": len(self.running), "waiting": len(self.waiting),
            "prefills": sum(1 for r in self.running if r.is_prefill_chunk),
            "starved_decodes": starved}) + "\n")
    return out
Scheduler.schedule = _logged_schedule
```

Treat the attribute names as a sketch to check against 0.32. `is_prefill_chunk` lives on `Request` and is updated after scheduling, so it lags by a step. Inter-call timestamps are also only an approximation of step time under async scheduling.

## INT8 W8A8 is the only untried lever big enough to matter

The kernels note makes a clear case. Weight-only formats (AWQ W4A16, FP8-Marlin W8A16) dequantize to BF16 and run BF16 MMA. At M ≈ 400-4500 they are compute-bound with extra dequant work, which matches E001 and E037. INT8 W8A8 runs CUTLASS `scaled_mm` on s8 MMA at about 2x the BF16 rate and halves weight bytes. vLLM 0.32 supports it on A100: `CompressedTensorsW8A8Int8.get_min_capability()` returns 75, with Cutlass, Triton and Humming INT8 kernels as candidates ([vllm compressed_tensors_w8a8_int8.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py); [vLLM INT8 docs](https://docs.vllm.ai/en/v0.8.5/features/quantization/int8.html)).

A recipe already exists for the base architecture. RedHat's Qwen3.5-9B W8A8 uses LLM Compressor GPTQ (`targets="Linear"`, `dampening_frac=0.01`). It ignores `lm_head`, `embed_tokens`, the vision tower and **all `linear_attn` (GDN) layers**, and it copies the MTP tensors back. It recovers **99.4-100.2%** of BF16 accuracy, e.g. GSM8k-Platinum 94.4 → 94.3, MMLU-Pro 82.4 → 82.0, AIME25 85.4 → 85.4, IFEval 89.5 → 89.6 ([RedHat Qwen3.5-9B W8A8](https://huggingface.co/RedHatAI/Qwen3.5-9B-quantized.w8a8)). That checkpoint is a derivative of base Qwen3.5, not MANGO, so the rules require us to rerun the recipe on `CMKL/MANGO1.5-Qwen3.5-9B` ourselves. The rules allow that.

Speed evidence is indirect, so treat the gain as a hypothesis. The only primary numbers are older-vLLM results: an 8B model async on an Ampere A6000 at **1.38x** QPS (2.2 vs 1.5), and a 70B on 4x A100 at **1.87x**. That paper's summary is that "W8A8 dominates in asynchronous continuous batching" ([Kurtic et al., "Give Me BF16 or Give Me Death"](https://arxiv.org/html/2411.02355v3)). There is no published A100 8B-class W8A8 vs BF16 measurement at batch 64-256. The team's coverage estimate runs as follows. MLP (about 4.8B params) plus full-attention projections (about 0.47B) move to INT8, while GDN projections (about 1.6B) and the LM head (about 1.0B) stay BF16. That puts **about 75% of body GEMM FLOPs on INT8**. If those GEMMs run 1.6-1.8x faster and GEMMs are 60-75% of peak step time, the whole step gets **about 1.2-1.35x faster**, for prefill chunks and decode verification alike. That is the only candidate in the notes that raises the binding prefill-capacity term. It also lowers energy per token, which pushes sE toward its cap, and it frees roughly 5 GB of weight memory for KV/state.

What INT8 could deliver depends on how a faster step translates into sT·sL. E023 is the only anchor: a ~14% compute cut halved sT·sL. Assume the response flattens on the way up, so that a 1.15-1.35x step speedup buys **+15% to +45% sT·sL**, and that sE rises to 0.97-1.0. Then **S lands at roughly 91-96, with a central estimate of 93-94** (e.g. +20% sT·sL and sE 0.99 gives 93.2; +30% gives 94.7). The upper end is plausible because peak misses are marginal, but it is still an extrapolation from a single official data point.

The serious risk is accuracy, and our measurement of it has a blind spot. AWQ failed on exactly the agent class (deterministic 2/20 vs BF16's 3-7/20), and the local agent checker (about 0.25) does not represent the official agentic accuracy of 0.82 ([KEY_FINDINGS.md](file:///home/researcher/repo/KEY_FINDINGS.md)). So the local final track (n=60 per class) can catch Thai/RAG/reasoning regressions but cannot clear the agent class. **One official val run should serve as the A gate before INT8 goes to an official final.** Current R = 1.034 gives some buffer. A second risk is MTP acceptance: quantizing the target shifts its distribution relative to the BF16 MTP head, and no source measures acceptance under W8A8 on a GDN hybrid. Acceptance must stay near 0.87/0.74, so record it from the server's spec-decode metrics as a go/no-go.

The build, as a sketch to adapt (LLM Compressor needs transformers ≥ 5 per its Qwen3.5 page ([llm-compressor Qwen3.5](https://docs.vllm.ai/projects/llm-compressor/en/latest/key-models/qwen3.5/))):

```python
# make_int8.py (run in a separate venv with llmcompressor + transformers>=5)
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor.modifiers.smoothquant import SmoothQuantModifier   # arm B only
SRC = "CMKL/MANGO1.5-Qwen3.5-9B"; OUT = "/home/researcher/models/MANGO1.5-W8A8"
ignore = ["re:.*lm_head", "re:.*embed_tokens$", "re:.*visual.*", "re:.*model.visual.*",
          "re:.*linear_attn.*"]               # arm C: drop linear_attn from ignore
recipe = [GPTQModifier(targets="Linear", scheme="W8A8", dampening_frac=0.01, ignore=ignore)]
# arm B: recipe = [SmoothQuantModifier(smoothing_strength=0.8)] + recipe
oneshot(model=SRC, dataset=calib_ds,          # ~512 in-domain chat-template samples:
        recipe=recipe, num_calibration_samples=512,   # Thai, RAG/long, reasoning traces,
        max_seq_length=8192, output_dir=OUT)          # tool-call turns
# then copy MTP tensors back from SRC (RedHat card: save_mtp_tensors_to_checkpoint(source_model=SRC, dest_dir=OUT))
```

One loading detail can silently break MTP. If vLLM builds the MTP layer under the checkpoint's compressed-tensors config, it may expect INT8 weights where we copied back BF16 tensors. So mirror the `quantization_config.ignore` list in RedHat's published `config.json` exactly; that checkpoint is known to serve with MTP. Treat "MTP head loads unquantized and acceptance holds" as an explicit gate, not an assumption.

```bash
# serve: identical to E030 except model paths and a pinned pool
THINK_BUDGET=3072 python ~/repo/tools/gray/vllm_think_budget.py serve /home/researcher/models/MANGO1.5-W8A8 \
  --served-model-name CMKL/MANGO1.5-Qwen3.5-9B --host 0.0.0.0 --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --language-model-only --max-model-len 41984 --kv-cache-memory-bytes 18640000000 \
  --max-num-seqs 256 --max-num-batched-tokens 4096 --long-prefill-token-threshold 1024 \
  --enable-chunked-prefill --no-enable-prefix-caching --mamba-ssm-cache-dtype bfloat16 \
  --speculative-config '{"model":"/home/researcher/models/MANGO1.5-W8A8","method":"mtp","num_speculative_tokens":2}'
# gate: log must show CutlassInt8ScaledMMLinearKernel; GDN must still report "GDN spec decode kernel: cuda"
```

INT8's first launch will compile cold, so warm it up with a throwaway launch before measuring. Run the first INT8 arm with the pool pinned at the champion's 17.36 GiB, so that it isolates the compute effect. A second arm can then give the freed ~5 GB to the pool. That memory reopens the question k=3 keeps failing on. Each extra MTP depth costs about 27 MiB of GDN state per sequence, about 3.4 GiB at 130 sequences, and E020/E032 collapsed for lack of exactly that ([speculative-decoding note's derivation from vllm/model_executor/layers/mamba/abstract.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/model_executor/layers/mamba/abstract.py)). k=3 adds about 0.59 accepted tokens per step at +33% verify cost, a trade that only pays off-peak. So the INT8 follow-up should be a k=3/k=2 batch-size schedule, not static k=3. Arm C, which also quantizes the GDN projections, lifts INT8 coverage to about 95% of body FLOPs. RedHat excluded those layers, plausibly because error accumulates in the recurrent state, so arm C needs its own accuracy gate.

## Cheap peak-only knobs can add a few points if the logs confirm them

**Batch-size-scheduled MTP depth** ships in 0.32 as `num_speculative_tokens_per_batch_size` inside `--speculative-config`. It takes inclusive (start, end, K) ranges starting at 1, clamped to the static `num_speculative_tokens`. The scheduler looks up K each step from the number of requests scheduled, prefill-chunk requests included ([vllm/config/speculative.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/config/speculative.py); [vLLM dynamic SD docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/dynamic_speculative_decoding/)). Because GDN slots are sized from the static max K, `[[1,96,2],[97,256,1]]` is memory-neutral. It keeps k=2 off-peak, which matters because static k=1 cost +23% J/req in E009. That run compiled cold on a 13.48 GiB pool, so the penalty is overstated, but the direction is plausible since most of the run is off-peak. The team's roofline arithmetic gives the trade. At peak, k=2 spends 3 verify tokens for about 2.61 accepted and k=1 spends 2 for about 1.87. If steps are compute-bound, k=1's TPOT is about 0.93x of k=2's and a third of the decode-verify compute is freed for prefill. If steps are memory-bound, k=2 wins by about 28%. E035's "decode-saturated" observation leans toward the compute-bound case. The expected S effect therefore spans **roughly -2% to +2%**. It is cheap to test but no path to 95 on its own. Correctness is the main risk. GDN metadata builds spec-state indices from the static K. MTP is not among the methods the docs name as tested (EAGLE, EAGLE-3, DFlash). Upstream still has open dynamic-SD fixes for autoregressive speculators, such as #48329 and #58821 ([GitHub PR search](https://api.github.com/search/issues?q=repo:vllm-project/vllm+%22dynamic+speculative+decoding%22+is:pr&per_page=15)). So run a greedy output-identity and accuracy smoke test against static k=2 before any timing run.

**Decodes-first ordering** is the Sarathi-Serve "stall-free" rule: ongoing decodes, then partial prefills, then new requests ([Sarathi-Serve](https://arxiv.org/pdf/2403.02310)). It is distinct from E040. E040 throttled long prefills to favour short *waiting* prompts, which took prefill capacity away from the long requests that then missed. Decodes-first costs prefill at most the ~390 decode tokens per step that decodes consume anyway. What changes is that late-admitted decodes stop starving. That should help the TPOT term (0.83) without starving long prefills. Run it only if the step log shows `starved_decodes > 0` on peak steps. It drops into the same wrapper, since `AsyncScheduler` inherits `Scheduler.schedule`:

```python
if os.environ.get("DECODE_FIRST") == "1":
    _prev = Scheduler.schedule
    def _decode_first(self, *a, **k):
        self.running.sort(key=lambda r: r.is_prefill_chunk)   # stable: decodes keep FCFS order
        return _prev(self, *a, **k)
    Scheduler.schedule = _decode_first
```

There is a real side effect. FCFS preemption evicts `self.running[-1]` ([vllm/v1/core/sched/scheduler.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py)), so long partial prefills become the preferred victims. With prefix caching off, a preempted 20K prompt loses all its computed work, and a hybrid also loses its recurrent state. Check the step log for preemptions, and add a small `--watermark` if they appear. Published gains for this family of ideas (Sarathi's 2.6x, SLOs-Serve's 1.34x burst resilience, SLOWeave's +39% goodput) are all measured against untuned baselines ([SLOs-Serve](https://arxiv.org/html/2504.08784v1); [SLOWeave](https://arxiv.org/abs/2609.07883)). The one paper that compared against a *tuned* static budget, P-PAS, found only about **3%** ([P-PAS](https://arxiv.org/abs/2608.15171)). That is the right prior for us. The more ambitious TPOT-slack-aware prefill cap and TTFT-deadline (EDF) ordering both reshuffle a fixed prefill capacity, which is what E040 showed loses. Keep them shelved unless INT8 creates surplus capacity to allocate.

**`--max-num-active-seqs`** (new in 0.32) caps admission into RUNNING while keeping `--max-num-seqs 256` for CUDA-graph capture ([vllm/config/scheduler.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/config/scheduler.py)). It is a cleaner version of E036's failed `--max-num-seqs 112`, but it moves along the same TTFT/TPOT trade-off. Since TTFT is already the weaker term, its prior is weak.

**Reasoning-length levers are nearly exhausted for this goal.** sE is at 0.96, and lower budgets have *hurt* sT officially (E039: sE 0.974 but sT 0.636). Two cheap server-side additions remain in the existing `to_sampling_params` wrapper. The first is native `repetition_detection` (exact tail n-gram, finish_reason `"repetition"`), as a guard against answer-phase loops. The second is per-class budgets, e.g. 1536-2048 when `tools` is present and 1024-1536 for very long prompts, but only after logs show those classes actually reach 3072 ([vllm/sampling_params.py](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/sampling_params.py); [ThinkBrake](https://arxiv.org/html/2510.00546v2)). Never use a zero budget for agent requests: ThinkLess-style immediate exit cut BFCL Live from 82.9 to 75.5 and Live-parallel to 37.5 ([ThinkBrake](https://arxiv.org/html/2510.00546v2)). If repetition detection is enabled, remap `"repetition"` to `"stop"` in the wrapper unless the grader is known to accept it. The upside is at most the sE gap (about +2.5% S if sE hits 1.0), and INT8 is likely to close that gap anyway.

## Run these experiments in this order, with these gates

Policy for spending official runs, from our calibration: a local edge of 12-16% has twice failed officially (E026, E033), while the structural budget change (+30% local) carried over strongly (+49% official) ([KEY_FINDINGS.md](file:///home/researcher/repo/KEY_FINDINGS.md)). So **every candidate gets at least 3 fresh-server local-final runs, each launched warm, with the pool pinned**. Only a structural change, or a local-final edge of at least 20%, earns an official run. INT8 must also pass an official val run as its accuracy gate.

| # | Experiment | Exact change vs E030 | Gate to proceed | Expected effect on S | Main risk |
|---|---|---|---|---|---|
| 0a | Warm-up launch + pin KV pool | Throwaway launch to `/health` and stop; then measured launch with `--kv-cache-memory-bytes 18640000000` (≈17.36 GiB) | Log shows no "Compiling a graph"; no OOM over 3 local finals | 0 on mean; makes all later A/B comparable | A cold start with the pin cannot fit (~40.8 GiB), so warm-up must come first |
| 0b | Step logging | `STEP_LOG=...` hook above | Quantify starved decodes, prefills/step, step time vs tokens at peak | None (diagnostic) | Slight CPU overhead |
| 1 | INT8 W8A8 (arm A) | Self-made GPTQ W8A8 MANGO, RedHat ignore list, MTP copied back, pool pinned at 17.36 GiB | Cutlass INT8 kernel in log; MTP head loads; acceptance ≈0.87/0.74; local-final accuracy within noise on all classes; 3 local finals | +3 to +9% (≈91-96, central 93-94), team estimate | Agent-class accuracy (needs official val); acceptance drop |
| 1b | INT8, official val | Same as 1 | Official A = 1.00 and agentic ≈0.82 | Gate only | Spends one official val run |
| 2 | INT8 + bigger pool | Arm A with `--kv-cache-memory-bytes` raised by the freed weight memory (re-derive from startup log) | Higher peak running count without TPOT loss | +0 to +2% | Over-admission raises TPOT (E033 pattern) |
| 3 | Dynamic MTP depth | `"num_speculative_tokens":2,"num_speculative_tokens_per_batch_size":[[1,96,2],[97,256,1]]`; sweep 64/96/128 | Greedy identity/accuracy vs static k=2; peak TPOT down on 3 local finals | -2 to +2% | GDN correctness with K < static max |
| 4 | Decodes-first | `DECODE_FIRST=1` patch | Only if 0b shows starved decodes; watch preemptions | 0 to +2% | Long prefills become preemption victims |
| 5 | INT8 + k=3/2 schedule | `"num_speculative_tokens":3,"num_speculative_tokens_per_batch_size":[[1,64,3],[65,256,2]]` on INT8 with enlarged pinned pool | No admission collapse on local final (E020/E032 failure mode) | 0 to +3% (sE, off-peak TPOT) | Memory collapse returns at peak |
| 6 | INT8 arm C | Also quantize `linear_attn` projections | Accuracy holds incl. long/RAG; acceptance holds | +1 to +3% over arm A | Error accumulation in recurrent state |
| 7 | Warm re-test of a cold-run rejection | E028 (`--max-num-batched-tokens 5120`) warm, pool pinned | ≥20% local edge only | Unknown, likely small | Noise masquerading as signal |

The table deliberately leaves out several ideas, each for a reason backed by evidence in the notes:

| Do not spend runs on | Why |
|---|---|
| Clock lock, power cap | Peak is compute-bound; SLO lost ~1:1 with energy (E012/E014/E022/E023, official 52.5 vs 59.0) |
| fp8 KV cache | Forces FlashInfer and downgraded CUDA graphs with spec decode; TPOT 60 ms (E021) |
| AWQ / Marlin W4A16, online FP8 W8A16 | Dequant-to-BF16 kernels slower at batch; agent accuracy loss (E001, E037) |
| EAGLE-3, DFlash, n-gram, suffix | Every drafter pays (K+1) GDN slots; DFlash block 8 ≈ 216 MiB/seq; EAGLE-3 acceptance 2.97 at 3 steps is no better than our MTP ([HF EAGLE-3](https://huggingface.co/yuyijiong/Qwen3.5-9B-Eagle3); [HF DFlash](https://huggingface.co/z-lab/Qwen3.5-9B-DFlash)); suffix is V1-only and needs arctic-inference |
| `-O3`, CUDA-graph capture size, attention backend, GDN env vars | `-O3` ≡ `-O2` in 0.32; explicit capture size drops the appended decode sizes; FLA Triton prefill and fused CUDA MTP decode are already the best SM80 paths |
| `--stream-interval` | Buffers tokens and distorts measured TTFT/TPOT ([vLLM Qwen3.5 blog](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)) |
| EDF / slack-aware prefill caps | Reshuffle fixed prefill capacity; E040 showed misses just move to long prompts |
| RecoverSSM backport (#58863 + #61309), Qwen3.5 fused-kernel PR #51012 | Stretch only. RecoverSSM would cut k=3 state from about 108 to about 30 MiB/seq, but it is an 18+-commit draft stack untested on A100 or `mamba_cache_mode none`, with non-identical greedy outputs ([PR #61309](https://github.com/vllm-project/vllm/pull/61309)). #51012 reports MTP decode steps 17-38% faster on the 9B, but on GB200 and unmerged ([PR #51012](https://github.com/vllm-project/vllm/pull/51012)) |

## Conclusion

The project has moved from a regime where better scheduling paid to one where only more arithmetic per second does. Each of the three big wins (bf16 SSM state, the per-request prefill cap, the thinking budget) removed waste. The remaining misses come from a GPU that runs at full compute for a few seconds at peak, and only a cheaper FLOP (INT8) or fewer FLOPs per accepted token (peak-time k=1) can address that. 95 therefore rests on one bet, INT8, which must land in the upper half of its range, plus a cluster of low single-digit refinements. Each refinement is individually invisible under ±10% official noise and is worth taking only once launches are warm and pinned and step logs confirm its mechanism.

The cold-compile finding also changes how to read the past. Our local A/B history was noisier than the grader alone would explain: every new-config first launch ran with about 22% less KV memory than the champion. That disadvantage hit exactly the memory-sensitive changes (MTP depth, batch budget, admission caps) we were trying to judge. The warm-up launch is the cheapest step on the list, and it is the prerequisite for the most important experiment. INT8's first launch will be cold, and without a warm-up its best result would be measured on a crippled server.
