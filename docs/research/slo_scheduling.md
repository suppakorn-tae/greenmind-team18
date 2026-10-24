# SLO-aware scheduling on one GPU with vLLM 0.32 (reducing prefill/decode interference, meeting TTFT and TPOT at peak)

Scope note: the strongest evidence here comes from reading the installed vLLM 0.32.0 source (`/home/researcher/vllm-env/lib/python3.12/site-packages/vllm`, commit g5df63a720), cited as `vllm/<path>:<line>`. Paper gains are all measured against untuned fixed-chunk or vanilla baselines. Our deployment already tunes the static knobs (4096 budget, 1024 per-request cap), so the marginal gain will be much smaller than the headline numbers. Repo notes (KEY_FINDINGS.md) put official run-to-run noise at about ±10%, and sT/sL enter the score with exponent 0.2.

## Q1. What do Sarathi-Serve, SOLA, SLOs-Serve, DynaServe, Aladdin, Llumnix, EDF/slack and adaptive-chunk papers propose for single-GPU colocated prefill/decode, and which exist in vLLM?

### Takeaway
Every single-GPU method comes down to two levers. (1) Put decodes first, then fill the rest of the step with prefill, sized so the step finishes before the tightest decode deadline (Sarathi "stall-free", SLOs-Serve, SLOWeave). (2) Order prefill work by TTFT deadline or slack instead of FCFS (SOLA, SLOs-Serve, FlowPrefill S-EDF). Stock vLLM v1 implements only a static version of (1): a global token budget plus a per-request cap. It does not order decodes before partial prefills, and it has no deadline awareness. Every paper system is a research fork. None ships as a vLLM `scheduler_cls` plugin.

### Cited Findings
**Sarathi-Serve (OSDI'24)**
- Builds each batch as all ongoing decodes, then partial prefills, then new requests. Prefill chunks are sized to the remaining token budget, which is derived from the TBT SLO ("stall-free batching"). — [Sarathi-Serve arXiv](https://arxiv.org/pdf/2403.02310); [OSDI'24 PDF](https://usenix.org/system/files/osdi24-agrawal.pdf)
- Reports up to 2.6x serving capacity under SLO for Mistral-7B on 1x A100 vs Orca/vLLM (2024-era vLLM without chunked prefill), and up to 3.7x for Yi-34B on 2x A100. Naive hybrid batching can inflate TBT up to 28.3x vs decode-only batches. — [Sarathi-Serve arXiv](https://arxiv.org/pdf/2403.02310); [OSDI'24](https://usenix.org/system/files/osdi24-agrawal.pdf)
- vLLM v1's chunked prefill with `max_num_batched_tokens` is the Sarathi idea, but the budget is static and set by the operator, not derived from an SLO. — vllm/config/scheduler.py:48-53, vllm/v1/core/sched/scheduler.py:611

**SLOs-Serve (CMU/Google, arXiv 2504.08784)**
- Fits a roofline-style batch-time model by regression on profiled runs (R² 0.82-0.93).
- Sets batch latency to the minimum TPOT among running requests. Decode tokens are allocated in EDF order: each scheduled token pushes that request's next deadline out by its TPOT. Leftover capacity goes to prefill chunks ordered by prefill (TTFT) deadline.
- Batch size is dynamic: the largest value that still meets all decode SLOs, rather than a fixed cap.
- A dynamic program does admission control. Requests whose SLO is unattainable (about 2%) go to a best-effort tier that runs on leftover budget and can be preempted. — [SLOs-Serve](https://arxiv.org/html/2504.08784v1)
- Built on vLLM as a backend (about 10K lines Python plus 1.5K C++). Planner calls take under 10 ms, mostly under 2 ms. — [SLOs-Serve](https://arxiv.org/html/2504.08784v1)
- Results: 2.2x geo-mean capacity at about 90% attainment over the best of Sarathi/vLLM; 1.76x over vLLM end-to-end on A100.
- Single-GPU ablations: SLO-adaptive speculative decoding 1.66x (2.01x ChatBot); burst-resilient scheduling 1.34x (1.66x Coder). Routing (1.19x) is multi-replica only. — [SLOs-Serve](https://arxiv.org/html/2504.08784v1)
- A later paper (Cascade, arXiv 2608.06557) argues that SLOs-Serve's combinatorial search limits online scalability. This is a third-party claim. — [Cascade](https://arxiv.org/pdf/2608.06557)

**SOLA (MLSys'25, Tsinghua/Infinigence)**
- Argues that coarse-grained scheduling creates a bias between TTFT and TPOT attainment and variance across requests. Proposes state-aware, per-iteration scheduling that balances the two.
- Reports attainment rising from 45.5% to 99.4%, and 1.04-1.27x more requests served under SLO vs SOTA. — [MLSys'25 SOLA page](https://mlsys.org/virtual/2025/3345)
- I could only read the abstract. The slides PDF could not be text-extracted here, so the exact mechanism (how it trades per-request TTFT vs TPOT slack) is unverified.

**SLOWeave / deadline-aware adaptive prefill chunking (arXiv 2609.07883, Sep 2026)**
- Picks the largest prefill chunk predicted to finish before the earliest active decode deadline, using a log-time search over a monotone iteration-cost model.
- +39% goodput (both TTFT and TPOT met) on mixed requests and +38% on long-context requests vs the best fixed chunk at a 25 ms TPOT target; 3.3x and 2.4x at 10 ms.
- Evaluated in a simulator plus an iteration-level GPU runtime. The abstract does not say it is vLLM-based. — [SLOWeave](https://arxiv.org/abs/2609.07883)

**P-PAS (arXiv 2608.15171, Aug/Oct 2026)**
- Switches vLLM's `max_num_batched_tokens` between a large budget under low prefill pressure and a smaller one when several prefills compete with active decodes.
- Gains are modest: P-PAS+ cuts mean E2E latency 3.0-3.1% vs the best static budget (Qwen3.8-27B, Nemotron-3.5-30B-A3B). TTFT/TPOT attainment not reported.
- Code is a vLLM fork: github.com/TimoSaemann/ppas-vllm. — [P-PAS](https://arxiv.org/abs/2608.15171)

**FlowPrefill (arXiv 2602.16603)**
- Replaces small fixed chunks with operator-level preemption of prefill, plus event-driven slack-aware EDF (S-EDF) ordering and SLO-aware batching.
- Built on vLLM 0.11.2. Up to 5.6x goodput, but mostly on 8x A800 disaggregated setups. In PD-colocation, TBT attainment improves up to 1.6x. — [FlowPrefill](https://arxiv.org/html/2602.16603v1)
- States that small fixed chunks cause throughput collapse (more kernel launches, repeated KV reloads, underused compute). This matches our finding that budgets of 2048-3072 lose. — [FlowPrefill](https://arxiv.org/html/2602.16603v1)

**TaiChi (arXiv 2508.01989)**
- Argues the best chunk size is the smallest one that still meets TTFT.
- Reallocates resources between requests at risk of missing TTFT or TPOT; up to 77% goodput gain claimed. — [TaiChi PDF](https://arxiv.org/pdf/2508.01989v1)
- I read search snippets only.

**Other systems (mostly not applicable to one GPU)**
- DuetServe (arXiv 2511.04791, ICML'26) runs aggregated by default. It switches on SM-level spatial multiplexing of prefill and decode on a single GPU when TBT degradation is predicted. — [DuetServe ICML'26](https://icml.cc/virtual/2026/poster/62917); [arXiv](https://arxiv.org/abs/2511.04791v1)
- DynaServe splits requests across cooperating GPU segments, with a per-GPU local scheduler for SLO batching. It is multi-GPU in essence. — [DynaServe](https://arxiv.org/pdf/2504.09285)
- Aladdin is cluster-level worker placement and scaling from prefill/decode latency models. — [Aladdin](https://arxiv.org/html/2405.06856v1)
- Llumnix (live migration across instances) needs more than one instance. I did not fetch it; it is excluded by our one-GPU, one-process constraint.

### Inferences
- The levers with good evidence that we have NOT already tried are both scheduling-order changes: (a) decodes first and (b) TTFT-deadline ordering of prefill work, plus (c) making the total prefill per step depend on decode slack. Static budget sweeps (P-PAS-style switching) give about 3% in the only paper that measured them against a tuned static budget. That fits our failed sweeps.
- Realistic expectation after tuning: single-digit to low-double-digit percentage-point attainment gains in the peak phase, not the 1.3-2x headline numbers. That is near official noise (±10%), so confirm on the local final track over several runs.

### Gaps
- SOLA's exact algorithm (full text not readable here).
- No public `scheduler_cls` plugin implementing EDF or SLO-aware budgeting for vLLM v1 0.2x/0.3x was found.
- SLOs-Serve and FlowPrefill code bases target older vLLM versions. Porting effort to 0.32 is unknown.
- DynaServe and DuetServe numbers not verified (abstracts only).

## Q2. Which vLLM v1 (0.32) scheduler options exist, what do they do, and how are they enabled?

### Takeaway
In 0.32 the only knobs that change per-step prefill/decode mixing are `max_num_batched_tokens`, `long_prefill_token_threshold` and `scheduler_cls`. `prefill_schedule_interval`, `throttle_prefills`, `prefill_capacity_bound` and `deferred_waiting` do nothing on our deployment. `long_prefill_token_threshold_adaptive` can only RAISE the cap, so at peak concurrency it is a no-op. `max_num_active_seqs`, `watermark` and `scheduler_reserve_full_isl` are admission knobs. `max_num_queued_tokens/reqs` reject with HTTP 503, which counts as an SLO failure for us.

### Cited Findings
**Prefill throttling and deferred waiting (no effect here)**
- `throttle_prefills` / `prefill_schedule_interval` / `prefill_capacity_bound` are a DP-only feature ("DP prefill balancing").
- Base `EngineCore._should_throttle_prefills()` always returns False ("Overridden by the DP engine core; never throttles otherwise"). Only `DPEngineCoreProc` overrides it, and that class asserts `model_config.is_moe`. — vllm/v1/engine/core.py:652-655, 2157-2162, 2318-2325; vllm/config/scheduler.py:203-206 ("For data-parallel deployments...")
- `deferred_waiting` is just the set of waiting requests skipped this step because they are blocked (e.g. waiting for KV transfer). It is not a throttle. — vllm/v1/core/sched/scheduler.py:228-230, 904-910

**Per-request prefill cap (`long_prefill_token_threshold`)**
- Applied to RUNNING requests (including in-progress partial prefills) and to newly admitted WAITING requests. — vllm/v1/core/sched/scheduler.py:708-709, 1155-1156
- Disabled when there is only one request in the system (`num_eligible_reqs > 1`). — scheduler.py:643-648
- The cap bounds each request, not the total. Four long prompts at 1024 each still fill a 4096 budget.

**`long_prefill_token_threshold_adaptive` (bool, default False)**
- Sets cap = max(cap, max_num_batched_tokens // (running + waiting)). — vllm/config/scheduler.py:86-90; scheduler.py:650-655
- With 4096 budget and N above 4 requests, 4096/N is below 1024, so it never changes anything at peak. It only enlarges the cap when 2-3 requests are present.

**`max_num_active_seqs` (new; `<= max_num_seqs`)**
- Limits admission into RUNNING while keeping `max_num_seqs` for runner buffers and CUDA-graph capture sizes. — vllm/config/scheduler.py:68-77; scheduler.py:132-136, 923
- This differs from our failed `--max-num-seqs 112` run, which also shrank graph capture.

**Request rejection (`max_num_queued_reqs` / `max_num_queued_tokens`)**
- Enforced in the API server process: when the in-flight count, or the total prompt tokens currently in prefill, exceeds the limit, new requests get HTTP 503.
- The docstring frames `max_num_queued_tokens = target_TTFT * prefill_throughput` as TTFT QoS. — vllm/config/scheduler.py:92-128

**Admission headroom knobs**
- `watermark` (fraction of KV blocks kept free at admission, default 0). — vllm/config/scheduler.py:193-198
- `scheduler_reserve_full_isl` (default True): checks that the full prompt fits in KV before admitting, to prevent over-admission and thrashing. — vllm/config/scheduler.py:186-191

**Scheduling policy**
- `policy` is only "fcfs" | "priority". Priority orders the waiting queue and preemption, not intra-step budget. — vllm/config/scheduler.py:21, 159-165; vllm/v1/core/sched/request_queue.py

**`scheduler_cls` (custom scheduler)**
- Accepts a class or "mod.Class" path. — vllm/config/scheduler.py:174-180, 228-253
- Warns: "If you have subclassed Scheduler instead of AsyncScheduler, you will see degraded performance due to async scheduling being disabled". A custom class therefore must subclass `vllm.v1.core.sched.async_scheduler.AsyncScheduler`. — vllm/config/scheduler.py:240-250
- Also warns that the interface is "not public and compatibility may not be maintained".

**Related history**
- V0-era `max_num_partial_prefills` / `max_long_partial_prefills` limited concurrent long partial prefills (described in community posts). They are not present in the 0.32 SchedulerConfig. — [DEV Community explainer](https://dev.to/ji_ai/chunked-prefill-why-one-long-prompt-stalls-every-decode-ln0); vllm/config/scheduler.py (absent)

**Hybrid-specific scheduler path**
- `_mamba_block_aligned_split` clips chunk ends to Mamba block boundaries, but only when `mamba_cache_mode == "align"`. The default is "none". — scheduler.py:377-379, 443-556; vllm/config/cache.py:189

**How the step is built (0.32)**
- `schedule()` first walks `self.running` in list order. That list is append-on-admission order and is never re-sorted, so decodes and in-progress partial prefills are interleaved by admission time.
- Each running request takes up to min(remaining tokens, cap, token_budget). The loop stops when `token_budget` hits 0.
- Only then are WAITING requests admitted with whatever budget is left, and only if no preemption happened this step. — scheduler.py:659-711, 856-858, 895-923, 1345
- Each scheduled request also consumes `draft_slots` (MTP) from `input_budget`. — scheduler.py:612-614, 857

### Inferences
- **Stall risk for late-admitted decodes (from code, not measured).** Running requests admitted after the long prompts (typically short requests that arrived during the burst) sit behind 3-4 long partial prefills in `self.running`. When those prefills consume the 4096 budget (4 × 1024), the loop exits and the later decodes get NO tokens that step. This is exactly the "decode stall" that Sarathi's decodes-first ordering avoids. It would hit short-output requests that arrived at peak, which matches the diagnosis (TPOT 51-72 ms).
- **TTFT for short arrivals.** New short requests get only the budget left after all running partial prefills. With 3 long prefills in flight, about 600 tokens remain; with 4, zero. Their first chunk slips by one or more mixed steps, consistent with marginal TTFT misses (0.52-0.88 s vs 0.51-0.65 s).
- **Step cost estimate (rough).** A 4096-token mixed step on a 9B dense-FLOP model costs about 2·9e9·4096 ≈ 7.4e13 FLOP, roughly 0.3 s on A100 at about 230 TFLOPS effective. That is about 10x a decode-only step. With MTP accepting about 2.5 tokens/step, a 50 ms TPOT allows about 125 ms per step on average, so a handful of mixed steps blow a short-output request's TPOT. This estimate is my own, not measured; verify with step-time logging.
- **Rejection is not an option.** `max_num_queued_tokens` 503s would turn marginal misses into hard failures (and likely accuracy-zero). Use only if the grader treats 503 retries leniently, which is unlikely.

### Gaps
- No upstream doc or PR found describing `long_prefill_token_threshold_adaptive`, `max_num_active_seqs` or `watermark` rationale and benchmarks; the source docstrings are the only reference.
- Exact `draft_slots` value for MTP k=2 not checked (`spec.max_num_new_slots_for_drafting`).

## Q3. TPOT-aware / slack-aware token budgets: how to limit long-prompt prefill only when short requests are decoding, how to apply it in vLLM 0.32, and what it buys

### Takeaway
Implement it as an `AsyncScheduler` subclass passed via `--scheduler-cls` (CLI flag confirmed at vllm/engine/arg_utils.py:1758; `--max-num-active-seqs` and `--long-prefill-token-threshold-adaptive` are also exposed, arg_utils.py:1725,1740). Three edits, in order of effort and expected value:
- **(A)** Stable-sort `self.running` so decodes come first (Sarathi stall-free). About 10 lines.
- **(B)** Deadline/EDF ordering, so waiting requests with tight TTFT deadlines take budget ahead of partial prefills that have seconds of slack. About 40 lines.
- **(C)** A dynamic total-prefill cap per step, computed from decode TPOT slack and a profiled step-time model (SLOs-Serve / SLOWeave rule). About 80-120 lines.

Published gains for (C)-type budgeting are +39% goodput (SLOWeave, 25 ms TPOT) and 1.34x burst-resilience (SLOs-Serve), both vs untuned baselines. For us expect far less. (A) is the cheapest experiment and directly targets the stall the code shows.

### Cited Findings
- SLOs-Serve allocates decode tokens EDF by per-request TPOT deadlines and gives prefill only the capacity left within the batch-latency target (min TPOT). Burst-resilient scheduling alone is worth 1.34x capacity (1.66x on Coder). — [SLOs-Serve](https://arxiv.org/html/2504.08784v1)
- SLOWeave sets the prefill chunk to the largest predicted to finish before the earliest active decode deadline. +39%/+38% goodput at 25 ms TPOT vs the best fixed chunk. — [SLOWeave](https://arxiv.org/abs/2609.07883)
- FlowPrefill uses slack-aware EDF (earliest deadline among requests with non-negative slack) for prefill ordering, with SLO-aware batching that adds requests only if predicted latency fits the head request's remaining time. — [FlowPrefill](https://arxiv.org/html/2602.16603v1)
- Sarathi-Serve orders decodes, then partial prefills, then new prefills, and sizes chunks from the TBT SLO. — [Sarathi-Serve](https://arxiv.org/pdf/2403.02310)
- P-PAS (switching only the global budget on prefill pressure) gains only about 3% mean latency vs the best static budget. Changing the total budget alone, without ordering, is weak. — [P-PAS](https://arxiv.org/abs/2608.15171)
- Scheduler hooks in 0.32:
  - `schedule()` builds the step from `self.running` (in order), then the waiting queue. — vllm/v1/core/sched/scheduler.py:591-1407
  - `request.is_prefill_chunk` is maintained in `_update_after_schedule`. — scheduler.py:1634
  - `scheduler_cls` must subclass `AsyncScheduler` to keep async scheduling. — vllm/config/scheduler.py:240-250

### Inferences
**(A) Decodes first (lowest risk, about 1 hour)**
- Sketch, inferred from the code:
  ```python
  # greenmind_sched.py on PYTHONPATH; --scheduler-cls greenmind_sched.DecodeFirstScheduler
  from vllm.v1.core.sched.async_scheduler import AsyncScheduler
  class DecodeFirstScheduler(AsyncScheduler):
      def schedule(self, throttle_prefills=False):
          # stable: decodes keep admission order, partial prefills move behind them
          self.running.sort(key=lambda r: r.is_prefill_chunk)
          return super().schedule(throttle_prefills)
  ```
- Effect: decodes are never starved by partial prefills. Prefills still use the leftover budget.
- Risk 1, preemption victims change. FCFS preemption takes `self.running[-1]` (vllm/v1/core/sched/scheduler.py:794-802, 834; PRIORITY policy uses max(priority, arrival_time) instead), so long partial prefills become the preferred victims. With prefix caching off, a preempted 20K partial prefill loses its computed work. Fixing this needs the preemption victim selection overridden, or a watermark.
- Risk 2: check that `is_prefill_chunk` is correct for newly admitted and resumed requests; it is set after scheduling.

**(B) TTFT-deadline (EDF) ordering of prefill work (about 2-3 hours)**
- Compute `deadline = arrival_time + 0.5 + 0.25 * prompt_len/1000` (`request.arrival_time`, `request.num_prompt_tokens`) and `slack = deadline - now - est_remaining_prefill_time`.
- A 20K prompt has a 5.5 s deadline; a 500-token chat about 0.63 s. Within the partial-prefill section of `running`, and between running partial prefills and urgent WAITING requests, serve the smallest slack first.
- Simplest implementation: when the waiting queue head (or any waiting request within the first K) has slack below one step time, temporarily cap running partial prefills (skip them, or set their share to 0) so the waiting request is admitted this step.
- Cleaner but larger: override the running loop to treat in-flight partial prefills and waiting prefills as a single deadline-sorted prefill pool after decodes. This means copying `schedule()` (about 800 lines) and editing it.
- Risks: long prompts can starve and miss their own (looser) TTFT if shorts keep arriving. Mitigate with slack-based rather than strict priority, i.e. long prefills jump ahead once their slack shrinks.

**(C) TPOT-slack-aware total prefill cap (about half a day including profiling)**
- Maintain each running decode's TPOT "debt": `allowed = 0.05 * (n_out - 1)` vs `elapsed = now - first_token_time`. TPOT is a whole-span average, so early slack can absorb a slow step, but short-output requests cannot.
- Let `min_slack` be the smallest decode slack. Choose `prefill_tokens_this_step = max P such that step_time(decode_tokens + P) <= target`, with target ≈ max(TPOT·accepted_tokens_per_step, f(min_slack)). Then distribute P across prefills (EDF from B).
- `step_time(n)` is a linear fit from step-time logging (≈ a + b·n; for the attention layers add c·n·context).
- When no decode is close to its deadline, keep the full 4096 budget. This is why it should beat the static 2048/3072 budgets, which pay the per-step overhead even when nobody needs protecting.
- Implement by overriding `schedule()` and clamping `token_budget` for prefill requests only. Either pre-compute the cap and apply it inside a copied running/waiting loop, or temporarily lower `self.max_num_scheduled_tokens` and the per-request cap before calling `super().schedule()` when the waiting queue is non-empty and decodes are near deadline.
- The temporary-override hack is crude because `token_budget` also covers decodes. Reserve decode tokens first: `cap = decode_tokens + P`.
- Risks:
  - Model error.
  - Prefill throughput falls at peak, so long prompts may then miss TTFT. The 65K-token burst needs about 5 s of pure prefill at about 13K tok/s (rough estimate from the step-cost inference in Q2), against 5.0-6.25 s TTFT budgets for 18-23K prompts. There is little room to defer long prefills, so (C) should only shave prefill on steps where decodes are near their deadline.
  - CUDA-graph or batch-shape changes. Mixed steps run eager or piecewise anyway.

**Order of experiments**
- Run A, then A+B, then A+B+C, each on the local final track over 3 or more runs, since official noise is about ±10%.
- Also cheap to try: `--max-num-active-seqs 112-128` with `--max-num-seqs 256`. This keeps graph capture while capping running sequences at peak, unlike our failed `--max-num-seqs 112`. Expect the same TTFT/TPOT trade, but possibly a better operating point.

### Gaps
- No paper measured decodes-first or EDF ordering *on top of* vLLM v1's per-request cap, so the gain for us is unquantified.
- Need our own per-step time logs, i.e. step time vs scheduled tokens and number of prefills, to confirm that the decode-starvation scenario actually happens. Verify by logging `num_scheduled_tokens` per request per step at peak.

## Q4. Scheduling specifics for hybrid (Gated-DeltaNet/Mamba + attention) models

### Takeaway
I found no published scheduling paper specific to hybrid models. In vLLM 0.32 the hybrid-specific scheduler behaviour is limited to: Mamba block-aligned chunk splitting (only with `mamba_cache_mode="align"`, i.e. prefix caching; off for us), a multi-module MTP prefill-lookahead reservation, and per-sequence recurrent state slots that scale with speculative k (from our own experiments).

### Cited Findings
- `need_mamba_block_aligned_split = has_mamba_layers and mamba_cache_mode == "align"`. The default `mamba_cache_mode` is "none", so with prefix caching off, prefill chunks are not clipped to block boundaries. — vllm/v1/core/sched/scheduler.py:377-379; vllm/config/cache.py:189
- `_reserve_prefill_lookahead`: with multi-module MTP, chunk ends within `num_prefill_lookahead` of the prompt end are avoided. — vllm/v1/core/sched/scheduler.py:751-755
- Repo finding: MTP on the GDN hybrid multiplies per-sequence recurrent state (k+1 slots). fp32 state capped concurrency at about 46 sequences; `--mamba-ssm-cache-dtype bfloat16` fixed it. — /home/researcher/repo/KEY_FINDINGS.md
- Hybrid prefix-caching research (Marconi shared-prefix junction checkpoints) is wired into the 0.32 scheduler but only matters with prefix caching, which gave a 3% hit rate on our workload. — vllm/v1/core/sched/scheduler.py:536-548; /home/researcher/repo/KEY_FINDINGS.md

### Inferences
- With 24 of 32 layers being linear-attention (GDN), per-chunk prefill cost is close to linear in chunk size and nearly independent of the position in the prompt. Only the 8 full-attention layers grow with context. So a step-time model `a + b·tokens (+ small c·tokens·ctx)` should fit better than for dense-attention models. That makes model-based budgeting (Q3-C) comparatively easy here.
- GDN state is fixed-size per sequence, so the memory pressure at 130-140 sequences comes from KV in 8 attention layers plus state slots × (k+1). Admission knobs (`max_num_active_seqs`, `watermark`) interact with this. Preemption of hybrid requests discards recurrent state, and with prefix caching off a preempted request must re-prefill entirely. That argues for avoiding preemption, e.g. a small watermark, rather than over-admitting.

### Gaps
- No benchmarks found for chunked-prefill or scheduling interference specifically on Qwen3.5-style GDN hybrids.
- No literature found on whether GDN chunked-prefill kernels have a larger fixed per-chunk overhead than attention, which would change the optimal chunk size. Measure locally.
