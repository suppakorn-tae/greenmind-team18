# Server-side reasoning-length control for a Qwen3.5-style thinking model in vLLM

Scope note: "local source" citations point to the vLLM 0.32.0 install at `/home/researcher/vllm-env/lib/python3.12/site-packages/vllm/`, which I read directly. vLLM `main` may differ. The MANGO model card was read from the local copy at `/home/researcher/models/MANGO1.5-Qwen3.5-9B/README.md`, which mirrors the HF card.

## 1. Accuracy vs. tokens for budget forcing, NoThinking, ThinkLess, early exit (DEER, confidence, answer convergence), adaptive budgets and loop detection, for Qwen3 8B-14B-class models

### Takeaway
Hard truncation at about half the model's natural reasoning length costs a lot on hard math. For Qwen3-8B, AIME24 drops from 73.3 to 43.3 and MATH-500 from 96.2 to 90.2. Dynamic early exit, where the model stops itself once a trial answer is confident, keeps accuracy and cuts 30-60% of tokens (Qwen3-14B DEER: 82.8 to 83.7 average). It needs extra trial-answer decoding, though, so it is not an hours-scale vLLM change. Our 3072 budget sits well above our 1200-1700-token class means, so it trims only the tail. The large drops in the literature describe budgets near or below the mean, which matches what we saw at 1536/2048.

### Cited Findings
**Budget forcing (hard cap + forced end-of-thinking)**
- s1 budget forcing ends thinking early by appending the end-of-thinking delimiter, optionally followed by "Final Answer:". It extends thinking by appending "Wait". Scaling s1-32B with budget forcing raised AIME24 from 50% to 57%. I read only the abstract, so per-cap accuracy numbers are not captured here. — [s1 abstract](https://arxiv.org/abs/2501.19393); delimiter description from [Budget Guidance paper](https://arxiv.org/html/2506.13752v1)
- **Qwen3-8B with budget forcing at about 50% of natural thinking length** (accuracy / avg thinking tokens). Source: Budget Guidance, Table 1. — [Li et al., "Steering LLM Thinking with Budget Guidance"](https://arxiv.org/html/2506.13752v1)

  | Benchmark | Full thinking | Budget forcing | Budget guidance |
  |---|---|---|---|
  | MATH-500 | 96.2 / 4613 | 90.2 / 2545 | 93.0 / 2062 |
  | AIME24 | 73.3 / 13660 | 43.3 / 4010 | 50.0 / 3981 |
  | AMC | 91.6 / 8740 | 77.1 / 3807 | 80.7 / 3869 |
  | OlympiadBench | 71.7 / 9424 | 61.6 / 3712 | 65.6 / 3639 |

- The same paper says budget forcing "may cut off unfinished thoughts and force premature answers". Under strict budgets it sometimes scores below NoThinking (e.g., MATH-500 on DeepSeek-R1-Distill 7B/32B). Budget guidance needs a trained auxiliary predictor, so it is not a drop-in. — [Budget Guidance](https://arxiv.org/html/2506.13752v1)
- Qwen3-235B-A22B shows "scalable and smooth performance improvements" as the thinking budget grows on 4 math/code/STEM benchmarks. Exact curve values are only in Figure 2 and not in the text. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- DART reports truncation (thinking never closes) on OlympiadBench with Qwen3-8B falling from 98.7% at a 2K budget to 2.3% at 32K. This comes from a search-result summary; I could not read the PDF to verify. — [DART, arXiv 2606.23181](https://arxiv.org/abs/2606.23181)

**NoThinking (skip reasoning with an empty or prefilled think block)**
- On DeepSeek-R1-Distill-Qwen, NoThinking beats Thinking at matched token counts on 7 reasoning datasets. The gap is largest at low budgets: 51.3 vs 28.9 on AMC23 at 700 tokens. Its advantage grows with pass@k. The best results use parallel sampling plus best-of-N or verifier selection. — [Ma et al., "Reasoning Models Can Be Effective Without Thinking"](https://arxiv.org/abs/2504.09858)
- Qwen3 thinking vs. non-thinking from the same checkpoint differs widely by task. Qwen3-8B: AIME'24 76.0 vs 29.1, AIME'25 67.3 vs 20.9, MMMLU 74.4 vs 66.9, INCLUDE 67.8 vs 62.5, PolyMath 42.7 vs 18.8, BFCL v3 68.1 vs 60.2. Qwen3-14B: AIME'24 79.3 vs 31.7, MMMLU 77.9 vs 72.6, BFCL v3 70.4 vs 61.5. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)

**ThinkLess (very early `</think>` plus an output-regulation instruction)**
- ThinkLess inserts `</think>` shortly after `<think>` and adds a short task-specific format instruction, e.g., "Give only the final numerical answer." Averaged over GSM8K/MMLU/GPQA/BBH: Qwen2.5-7B-based distill 62.28 / 2189 tokens vs ThinkLess 62.91 / 394; Qwen2.5-14B-based 74.82 / 2145 vs 73.07 / 374. The authors attribute most of the early-cut accuracy loss to malformed output rather than missing reasoning. Accuracy against cut position is U-shaped. The paper contains internal number inconsistencies. Qwen3 was not tested. — [ThinkLess, arXiv 2505.15684](https://arxiv.org/html/2505.15684v2)

**Early exit (DEER: confidence on a trial answer at "Wait" transition points)**
- DEER watches transition points such as "Wait" and induces a trial answer with a `\boxed{}` answer prompt. It exits when the geometric-mean max-token probability of the answer is ≥ 0.95, and is robust over 0.90-0.97. Across 11 models and 10 benchmarks, CoT length drops 19.1-80.1% and accuracy rises 0.3-5.0%. — [DEER, arXiv 2504.15895](https://arxiv.org/html/2504.15895v3)
- **Qwen3-14B with DEER** (accuracy, tokens; tokens as % of vanilla): GSM8K 95.1/2047 → 95.3/840 (41%); MATH-500 93.8/4508 → 94.0/3074 (68%); AMC23 95.0/7139 → 95.0/4763 (67%); AIME24 70.0/10859 → 76.7/7619 (70%); GPQA-D 60.1/7339 → 57.6/2898 (39.5%, −2.5 pt). The average goes from 82.8 to 83.7. Code compression averages only 19.9%, vs 61.5% for math/science. Only Qwen3-14B appears in the main-table excerpt I read. Accuracy results used vLLM. Latency was measured in HF Transformers: −28% to −40% for base DEER. — [DEER](https://arxiv.org/html/2504.15895v3)

**Answer-convergence and oracle analyses**
- On math tasks, models usually settle on their final answer after about 60% of reasoning steps. Consistency-based early stopping cut tokens by more than 40% on NaturalQuestions and improved accuracy. — [Answer Convergence as a Signal for Early Stopping, arXiv 2506.02536](https://arxiv.org/html/2506.02536v1)
- Oracle Pareto frontiers across 6 reasoning models (Qwen3-4B/30B-A3B-Thinking-2507, R1-Distill 7/14/32B, GPT-OSS-20B) give the relative budget needed to match full-thinking accuracy: routing (think/no-think) needs 71.4%, a fixed per-question budget 35.0%, and dynamic early exit 14.2%. The deployable public methods fall below these frontiers. — [Pareto Frontiers of Reasoning Termination, arXiv 2610.13570](https://arxiv.org/html/2610.13570)
- A critique: answer-level confidence or consistency signals measure "answer readiness" rather than reasoning completion, and can exit prematurely. — [arXiv 2605.17672 (via EmergentMind)](https://www.emergentmind.com/papers/2605.17672)

**Adaptive (per-query) budgets**
- DART (training-free) samples two cheap no-think drafts. If they agree, it answers directly. Otherwise it predicts a thinking budget from draft entropy. It reports 32-73% fewer thinking tokens while preserving or improving always-think accuracy in most settings. — [DART abstract](https://arxiv.org/abs/2606.23181)

**Looping / repetition**
- Reasoning models loop more at low temperature. Larger models loop less, and distilled students loop more than their teachers. On the paper's synthetic task, greedy decoding gave a loop count of 11.84 with 44.43% accuracy, vs about 2.1 loops and about 99% accuracy at positive temperature. — [Pipis et al., "Wait, Wait, Wait... Why Do Reasoning Models Loop?" (ICML 2026)](https://arxiv.org/abs/2512.12895v1)
- Qwen docs advise against greedy decoding for thinking models: it "can lead to performance degradation and endless repetitions". — [Qwen quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)
- Base Qwen3.5-9B with thinking on looped "through repeated draft translations of a single sentence", hitting a 2,048-token cap still undecided. The MANGO1.5 fine-tune closes the same task in under 750 tokens. — [MANGO1.5 model card](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B)
- Across samples, long natural reasoning correlates with wrong answers. R1-32B accuracy is 71.9% for traces under 4K tokens and 44.7% above 12K (correlational, not causal). — [When More Thinking Hurts, arXiv 2604.10739](https://arxiv.org/html/2604.10739v1)

### Inferences
- Our setup differs from the literature's truncation experiments. The 3072 budget is about 1.8-2.5× our reasoning-class mean, while Budget Guidance cut at about 0.5× the natural length. The 43-point AIME drop does not predict our accuracy at 3072. It does explain why 1536/2048 (about 1.0-1.3× mean) started forcing answers on genuinely unfinished reasoning.
- Requests running to 8192 are probably dominated by loops, not productive reasoning. Greedy decoding, temperature 0 on a 9B fine-tune, is the documented loop trigger, and long traces correlate with wrong answers. So capping or killing loops gains score at near-zero accuracy cost. This matches our 59 → 88 result.
- DEER, answer convergence and DART all need extra forward passes per transition point (trial-answer decoding or drafts), plus orchestration outside a single sampling stream. None is natively in vLLM. Implementing one would take days, not hours, and the extra decode work competes with our TPOT SLO. A ThinkBrake-style single-stream logit-margin check (Q3) is the closest hours-scale approximation of early exit.
- NoThinking or very early cuts (ThinkLess) need a task-specific format instruction to keep accuracy. We cannot rely on the client prompt for that, and Qwen3 non-thinking loses about 40-50 points on AIME. So per-class zero-budget is plausible only for classes where thinking adds nothing (see Q4).

### Gaps
- No source gives a Qwen3-8B/14B accuracy-vs-budget curve at budgets around 2K-4K tokens on chat/RAG-style tasks. The literature is math-heavy and uses 4K-32K natural lengths.
- s1 per-cap numbers and the Qwen3 Figure 2 values were not obtainable from the text I read. DART internals were unreadable (PDF). DEER per-model Qwen3-8B numbers were not in the excerpt.
- No source measures greedy (temperature 0) budget-forcing accuracy specifically. Most papers sample at temperature 0.6.

## 2. Qwen's recommended way to apply a thinking budget, transition phrases, and observed accuracy vs. budget

### Takeaway
Qwen's official method is two-pass generation: generate up to the budget, and if `</think>` has not appeared, append the stop phrase "Considering the limited time by the user, I have to give the solution based on the thinking directly now." plus `</think>`, then continue. Qwen says this ability was not explicitly trained but emerges from Thinking Mode Fusion. The docs recommend budgets above 1024. The Qwen3.5 model card does not mention a thinking budget at all, so the behaviour on Qwen3.5 derivatives is unvalidated by Qwen.

### Cited Findings
- Qwen docs: the thinking budget is officially exposed only in Alibaba Cloud Model Studio. With open-source frameworks, generate twice: first up to the budget, then, if thinking is unfinished, append the early-stopping prompt and continue. Their example uses a toy budget of 16. They recommend tuning the budget to acceptable latency and setting it "above 1024 for meaningful gains". — [Qwen quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)
- Official stop phrase: "Considering the limited time by the user, I have to give the solution based on the thinking directly now." followed by `</think>`. The Qwen3 report says the model then answers from its partial reasoning, and that this ability "is not explicitly trained but emerges naturally as a result of applying Thinking Mode Fusion". — [Qwen quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html); [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- The Qwen3 report's evaluation uses a 32,768-token output limit and an 8,192-token thinking budget in thinking mode for some evaluations (from a search-result summary of the PDF). — [Qwen3 Technical Report PDF](https://arxiv.org/pdf/2505.09388)
- Accuracy vs budget: Qwen3-235B-A22B shows smooth, monotone gains with budget on 4 math/code/STEM benchmarks (Figure 2). Qwen expects further gains past 32K. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- Qwen3.5-9B model card: thinking-general sampling is temperature 1.0, top_p 0.95, top_k 20, presence_penalty 1.5. Precise coding uses temperature 0.6 with presence_penalty 0.0. presence_penalty 0-2 reduces endless repetition, but "higher values may cause occasional language mixing and a slight drop in performance". Recommended output length is 32,768, or 81,920 for competition math/code. No thinking-budget guidance is given. The /think and /nothink soft switches are not supported on Qwen3.5. Thinking is disabled via `chat_template_kwargs: {"enable_thinking": false}`. — [Qwen/Qwen3.5-9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- The MANGO1.5 chat template (local copy) emits `<think>\n` at the generation prompt. With `enable_thinking=false` it emits `<think>\n\n</think>\n\n`. Reasoning from earlier assistant turns is kept only for messages after the last user query, i.e., within the current tool-call chain; older turns are stripped. — [MANGO1.5 model card/repo](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B) (template lines 89-104, 149-152 of local `chat_template.jinja`)
- vLLM's docs show a transition phrase inside `reasoning_end_str` for Qwen3: `"I have to give the solution based on the reasoning directly now.</think>"`. The offline example uses "...based on the thinking directly now.</think>". — [vLLM reasoning outputs docs](https://docs.vllm.ai/en/latest/features/reasoning_outputs/)
- Other ecosystems use similar forced messages. A closed vLLM PR (#37112) proposed a `reasoning_budget` that force-injects "Reasoning budget exceeded, need to answer." plus the end token. — [vLLM PR #37112](https://github.com/vllm-project/vllm/pull/37112)
- Pitfall: in SGLang, `thinking_budget` was not enforced for Qwen3.6 because the thinking token IDs were hardcoded and did not match the tokenizer. With budget 200 the model produced about 1,400 reasoning tokens. — [sglang issue #25536](https://github.com/sgl-project/sglang/issues/25536)

### Inferences
- Our transition phrase ("Considering the limited time, I have to stop thinking and give the final answer directly now.\n</think>\n\n") is a close paraphrase of Qwen's official one, so phrase choice is unlikely to be a lever. Qwen's own wording ("give the solution based on the thinking") may anchor the answer to the partial reasoning slightly better than "give the final answer directly". At most this is a cheap A/B.
- Qwen's guidance assumes temperature 0.6-1.0 with presence_penalty. Our greedy setting is outside Qwen's recommended regime, so the loop/runaway behaviour after forced `</think>` at low budgets is expected rather than a vLLM bug.
- Because forced termination is emergent rather than trained, its reliability on a fine-tune (MANGO1.5) is not guaranteed. Our observation that the model sometimes keeps reasoning inside the answer after a forced `</think>` at 1536/2048 fits this.

### Gaps
- No official Qwen3.5 thinking-budget doc, recommended phrase, or accuracy-vs-budget curve for 8-9B models was found. The Qwen3 Figure 2 values are not in the text.
- I did not verify the exact whitespace or punctuation around `</think>` in Qwen's docs code snippet.

## 3. Server-side ways to stop runaway generation after `</think>` (answer cap, repetition handling, loop detection) and conflicts with client-fixed max_tokens/temperature

### Takeaway
vLLM 0.32.0 has two native levers. The first is per-request `thinking_token_budget`, which applies only inside reasoning. The second is per-request `repetition_detection`, an exact tail n-gram detector checked in the scheduler's stop check that ends the request with finish_reason `"repetition"`. Both can be defaulted server-side in our existing `to_sampling_params` wrapper. An answer-phase token cap, or a penalty applied only after `</think>`, is not native. It needs a custom logits processor, and our server runs Model Runner V2, so the processor must use the V2 interface. Server-side overrides of these fields do not conflict with client `max_tokens`/`temperature`, because they are separate fields. `presence_penalty` defaults also apply under greedy decoding when the client does not send them.

### Cited Findings
**`thinking_token_budget` mechanics (local source, v0.32.0)**
- `ReasoningConfig` has `reasoning_start_str` and `reasoning_end_str` (the string forced when the budget is exhausted, which may include a transition phrase). It also derives `natural_reasoning_end_token_ids` from the parser's end token (`</think>`) separately. — [vllm/config/reasoning.py (local 0.32.0)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/config/reasoning.py)
- Our server runs Model Runner V2. Every budget experiment's server log (`/home/researcher/repo/experiments/E027`…`E038/server*.log`, including E030 = champion) prints "Using V2 Model Runner". In 0.32.0, V2 is the default unless `_get_v2_model_runner_unsupported_features` lists something. Its speculative blocklist is only `suffix`, `medusa`, `mlp_speculator`, `custom_class`, plus parallel-drafting EAGLE and heterogeneous-vocab drafts, so `mtp` stays on V2 (vllm/config/vllm.py lines ~703-739 and ~2927-2934). — [vllm/config/vllm.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/config/vllm.py)
- V2 runner budget kernel: per request, it finds the last `reasoning_start` (`<think>`, which for our template sits in the prompt) and the last **natural** end (`</think>`) in committed plus draft tokens. If the last start is after the last end, it counts reasoning tokens. Once the budget is exhausted, it forces the `reasoning_end_str` tokens one by one, and forcing wins over custom logits processors. After a natural or forced `</think>`, nothing limits the answer. If the model emits a new `<think>` in the answer, a fresh budget applies. It is speculative-decoding aware. — [vllm/v1/worker/gpu/sample/thinking_budget.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/thinking_budget.py); [V2 logits-processor interface docstring (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/logits_processor/interface.py)
- Caveat for the **V1** runner path (`vllm/v1/sample/thinking_budget_state.py`): it detects natural end using `think_end_token_ids = reasoning_end_token_ids`, i.e., the whole forced string. When a transition phrase is configured, a naturally emitted bare `</think>` may not be recognised, so the count could continue into the answer and inject the phrase mid-answer. This is an inference from reading the source, not a confirmed bug. It does not affect us while V2 is in use, but it matters if a flag ever forces V1. — [vllm/v1/sample/thinking_budget_state.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/sample/thinking_budget_state.py)
- `thinking_token_budget` and `repetition_detection` are first-class fields on `ChatCompletionRequest` and are passed into `SamplingParams`. An older report (vLLM 0.18) found the budget silently ignored when passed via `extra_args`. — [chat_completion/protocol.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py); [vLLM forum](https://discuss.vllm.ai/t/thinking-token-budget-silently-ignored-when-passed-via-extra-args-in-vllm-0-18-0/2533)

**Native repetition detection (local source)**
- `RepetitionDetectionParams(max_pattern_size, min_pattern_size, min_count≥2)` checks whether the output tail is the same N-gram repeated `min_count` times, for each N in [min, max]. It runs in `check_stop` after `min_tokens`, sets status `FINISHED_REPETITION`, stop_reason `"repetition_detected"`, and finish_reason string `"repetition"`. The check runs over all output tokens, both reasoning and answer, and only matches **exact consecutive** repeats. — [vllm/sampling_params.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/sampling_params.py); [vllm/v1/core/sched/utils.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/core/sched/utils.py); [vllm/v1/engine/__init__.py FINISH_REASON_STRINGS (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/engine/__init__.py)

**Generation-config defaults vs. client params (local source)**
- `--override-generation-config` / `generation_config.json` can default only `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `temperature`, `top_k`, `top_p`, `min_p` and `max_new_tokens`. Fields the client sends win. `thinking_token_budget` and `repetition_detection` cannot be set this way, so they need the request wrapper. — [vllm/config/model.py `get_diff_sampling_param` (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/config/model.py); [chat protocol `_DEFAULT_SAMPLING_PARAMS` (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py)
- Qwen3.5: presence_penalty 1.5 is the recommended thinking-general value, but higher values risk language mixing and a slight performance drop. — [Qwen3.5-9B card](https://huggingface.co/Qwen/Qwen3.5-9B)

**Custom logits processors (local source)**
- `--logits-processors` takes FQCNs. Under Model Runner V2, classes must implement the V2 `LogitsProcessor` interface: `__init__(vllm_config, req_states)`, `add_request(req_idx, sampling_params) -> bool`, `apply_staged_writes()`, and `apply(logits, ctx)`. They read committed tokens from `req_states.all_token_ids` up to `total_len`, and this step's draft tokens from `ctx.input_ids`, with one row per draft token under spec decoding. `apply()` runs after penalties/grammar and before temperature/top-k. Thinking-budget forcing runs after it and overrides it. — [V2 interface.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/logits_processor/interface.py); [V2 loader.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/logits_processor/loader.py); [V2 sampler.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/sampler.py)
- Under the V1 runner, the rejection sampler applies only `non_argmax_invariant` processors. A V1 processor that changes the argmax must return `is_argmax_invariant() = False`. — [vllm/v1/sample/rejection_sampler.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/sample/rejection_sampler.py)

**Early-exit heuristic that fits a single stream (ThinkBrake)**
- ThinkBrake: at each sentence boundary inside thinking, compute the log-prob margin between the top token and `</think>`, and stop thinking if the margin is ≤ 0.25. This is training-free and needs no extra passes. On Qwen3-4B-Thinking-2507 BFCL, Non-Live accuracy went 89.6 → 89.8 with 24.9% fewer tokens, and Live 82.9 → 82.4 with 15.3% fewer tokens. It was also tested on Qwen3-1.7B and Qwen3-8B. — [ThinkBrake, arXiv 2510.00546](https://arxiv.org/html/2510.00546v2)

### Inferences
- **Native `repetition_detection` is the quickest runaway guard.** Example default: `{"min_pattern_size": 8, "max_pattern_size": 200, "min_count": 3}`, tuned on logs from runaway requests. It cannot catch "brute-force enumeration" that never repeats exactly. The request ends early, so it no longer hits 8192 or burns energy, but the answer is whatever was emitted. That helps when the answer was already written before the loop began, and turns into a shorter wrong answer otherwise.
- **Check `finish_reason: "repetition"` with the client and scorer.** It is non-standard for OpenAI clients. If the scorer only accepts stop/length, have the wrapper remap it to `"stop"`. Also measure CPU cost: the check is pure Python per step per request, up to `max_pattern_size` × `min_count` comparisons, at about 200 seqs × 100+ tok/s each.
- **A loop or answer guard as a V2 logits processor is hours-scale.** It would detect `</think>` in `all_token_ids` per slot, then:
  - (a) after N answer tokens (e.g., 2048-3000; chat answers average ~600), boost `<|im_end|>` at the next newline or sentence boundary rather than hard-cutting mid-token; or
  - (b) on detecting an n-gram repeat in the answer, force `<|im_end|>`; or
  - (c) apply a presence or frequency penalty only to answer-phase rows.

  Risks: long code or long-context answers hit by (a); MTP draft rows need per-row handling; tool-call JSON must not be cut. Exclude requests with `tools` from (a), or end only after a complete `</tool_call>`.
- **Global `presence_penalty` via `--override-generation-config` is a one-flag experiment.** It reduces loops under greedy decoding when the client omits the field. But it touches every request, including Thai, tool-call JSON with repeated keys, and code, and Qwen warns about language mixing. Treat it as higher risk than the logits-processor scoping in (c).
- **None of these conflict with client `max_tokens=8192` or `temperature=0`.** They are separate fields (thinking/repetition), or only take effect when the client omits them (presence_penalty), or act on logits under greedy decoding.

### Gaps
- No measurement found of `repetition_detection` overhead or accuracy effect in vLLM, and no public benchmark of answer-phase caps.
- Whether the official scorer accepts finish_reason `"repetition"` is unknown.
- ThinkBrake's sentence-boundary margin check has not been evaluated on chat/RAG or Thai, and there is no public vLLM implementation that I found.

## 4. How reasoning truncation affects multilingual (Thai) and tool-calling accuracy

### Takeaway
Tool calling: zeroing reasoning hurts badly. Forced immediate `</think>` dropped BFCL Live from 82.9 to 75.5 and Live parallel to 37.5. Qwen3-8B thinking beats non-thinking on BFCL v3 (68.1 vs 60.2). Yet small models also overthink: an oracle early stop lifts BFCL from 85.8 to 94.2 with about 87% fewer tokens. So moderate, not zero, budgets fit agentic requests. Thai: for MANGO1.5 itself, thinking helps exams and NLU strongly (ThaiExam +24.8 pt) but is neutral or harmful for translation, QA and XNLI (measured greedy). No source measures Thai truncation directly.

### Cited Findings
**Tool calling**
- ThinkBrake oracle: inserting `</think>` at the best sentence boundary raises average BFCL accuracy from 85.8% to 94.2% and cuts tokens by about 87%. Models often reach the right call, keep reasoning, and then overwrite it with a wrong one. — [ThinkBrake](https://arxiv.org/html/2510.00546v2)
- Baselines on Qwen3-4B-Thinking-2507 BFCL:
  - ThinkLess (immediate `</think>`): Non-Live 87.7, but Live 75.5 (vs 82.9 base) and Live parallel 37.5.
  - Confidence of `<tool_call>` (Halt-CoT style): Non-Live 82.0.
  - NoWait: 87.9.
  - Earlier early-termination methods lost 4-50% on parallel-call categories (per the paper's summary). — [ThinkBrake](https://arxiv.org/html/2510.00546v2)
- Qwen3 thinking vs non-thinking, BFCL v3: 8B 68.1 vs 60.2; 14B 70.4 vs 61.5. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)

**Multilingual / Thai**
- MANGO1.5 (the served model), greedy decoding, 16,384-token context, think vs no-think:

  | Benchmark | Think | No-think |
  |---|---|---|
  | ThaiExam | 60.0% | 35.2% |
  | M3Exam-Thai | 65.4% | 55.4% |
  | Belebele | 80.0% | 72.1% |
  | XCOPA | 86.6% | 73.0% |
  | Wisesight | 45.5% | 37.6% |
  | XLSum ROUGE-L | 0.0452 | 0.0260 |
  | XNLI | 54.7% | 57.7% |
  | Flores200 BLEU | 34.1 | 37.0 |
  | iApp Wiki QA F1 | 67.6% | 69.0% |
  | Normalized average | 0.444 | 0.405 |

  The authors say reasoning helps multiple-choice/classification much more than open-ended generation. — [MANGO1.5 model card](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B)
- Qwen3-8B thinking vs non-thinking on multilingual: MMMLU 74.4 vs 66.9, INCLUDE 67.8 vs 62.5, PolyMath 42.7 vs 18.8. — [Qwen3 Technical Report](https://arxiv.org/html/2505.09388v1)
- Reasoning-model MGSM accuracy is highest with English reasoning and falls for lower-resource languages, Thai included. The gap widens as resources shrink. — [The Reasoning Lingua Franca, arXiv 2510.20647](https://arxiv.org/pdf/2510.20647)
- Reasoning in the non-English target language cut accuracy by about 29% for a 32B model. A 7B model reasoning in English beat a 32B model reasoning in the target language. — [Long CoT Reasoning Across Languages, arXiv 2508.14828](https://arxiv.org/html/2508.14828v2)
- Qwen3-4B-Thinking-2507 scores about 71% on Thai reasoning vs about 81% on English (different benchmark). — [Native Multilingual CoT in SEA languages, arXiv 2608.00533](https://arxiv.org/html/2608.00533)
- Base Qwen3.5-9B loops on Thai translation with thinking on, as noted in Q1. — [MANGO1.5 model card](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B)

### Inferences
- Thai script tokenizes into more tokens per idea than English. If MANGO reasons in Thai for Thai prompts, a fixed token budget truncates Thai reasoning at an earlier "semantic" point. I did not verify the reasoning language MANGO uses; check our logs.
- Thai generation-type requests (translation, summarization, extractive QA) look safe to give a low budget: no-think is equal or better on Flores, iApp QA and XNLI. Thai exam/MCQ needs thinking but rarely needs long thinking. So a 3072 budget is very unlikely to hurt Thai unless those requests are among the runaways.
- For agentic/tool requests, keep thinking on with a budget of roughly 1024-2048. Our agentic reasoning is already short, so a budget at or above its p95 adds safety without truncating. Never use 0: ThinkLess-style immediate exit hurt parallel calls badly.

### Gaps
- No study directly measures accuracy vs thinking budget for Thai, or for multi-turn tool calling (BFCL multi-turn, tau-bench) on Qwen3-class models. ThinkBrake covers single-turn only.
- The MANGO1.5 table compares thinking on vs off, not budgets, and its NLG metrics are low-signal (ROUGE-L 0.045).

## 5. Setting per-request budgets adaptively server-side (tools present, prompt length, language) with vLLM hooks

### Takeaway
Yes. The simplest and most robust hook is the one already in use: monkeypatching `ChatCompletionRequest.to_sampling_params`. It has the full request (`messages`, `tools`, `tool_choice`, and so on) and can set `thinking_token_budget` and `repetition_detection` per request from features such as tools present, number of turns, total characters, or Thai-script share. vLLM has no built-in difficulty-adaptive budgeting. Literature methods for difficulty prediction (DART, predictive scheduling) need extra drafts or trained predictors. The oracle analysis suggests per-question fixed budgets can match full accuracy at about 35% of tokens, if the predictor is good.

### Cited Findings
- Our wrapper (`/home/researcher/repo/tools/gray/vllm_think_budget.py`) patches `protocol.ChatCompletionRequest.to_sampling_params` to set `thinking_token_budget = THINK_BUDGET` when it is unset. `ChatCompletionRequest` carries `thinking_token_budget` and `repetition_detection` as first-class fields, and both are copied into `SamplingParams`. — local wrapper; [chat_completion/protocol.py (local 0.32.0)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py)
- V2 custom logits processors receive `sampling_params` at `add_request`. They can use `extra_args`, which the request's `vllm_xargs` populate (protocol.py line ~746: `extra_args = self.vllm_xargs or {}`), or which the wrapper can set on the returned `SamplingParams` directly. So per-request knobs for an answer-phase guard travel the same way. — [V2 interface.py (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/v1/worker/gpu/sample/logits_processor/interface.py); [protocol.py `vllm_xargs` (local)](file:///home/researcher/vllm-env/lib/python3.12/site-packages/vllm/entrypoints/openai/chat_completion/protocol.py)
- Oracle frontiers rank routing (think/no-think) < fixed per-question budget < dynamic early exit. To match full accuracy they need relative budgets of 71.4%, 35.0% and 14.2% respectively. — [arXiv 2610.13570](https://arxiv.org/html/2610.13570)
- DART: agreement between two no-think drafts means answer directly; otherwise the budget comes from draft entropy. 32-73% fewer thinking tokens. — [DART](https://arxiv.org/abs/2606.23181)
- The Qwen docs frame the budget as a latency/quality knob to set per use case ("above 1024 for meaningful gains"). — [Qwen quickstart](https://qwen.readthedocs.io/en/latest/getting_started/quickstart.html)

### Inferences
- **Feasible rule-based policy in under an hour** (all features are available in `to_sampling_params`):

  | Request type | Signal | Budget |
  |---|---|---|
  | Agentic | `tools` non-empty, or a `tool` role message in history | 1536-2048, at or above observed p95 |
  | Long-context/RAG | total message chars > ~20-30K | ~1024-1536; answers are grounded and reasoning is short |
  | Thai | Thai-script share of user text (regex `[฀-๿]`) | about the same as chat |
  | Math/code | keyword or LaTeX/code-fence heuristics | keep 3072 |
  | Everything else (default) | none | 3072 |

  Add `repetition_detection` to all requests, as in Q3.
- **Which policy wins:** a cheaper policy only beats uniform 3072 if those classes actually hit 3072. Our class means are 1200-1700, so most energy savings at 3072 already came from runaway elimination. Per-class tightening mainly pays off on the per-class tail. Check the per-class distribution of reasoning lengths (p90/p99) and forced-end counts in current logs before spending official runs.
- **Prompt token length:** it is not computed yet inside `to_sampling_params`. Character counts are a sufficient proxy. Tokenizing in the wrapper costs CPU per request.
- **Accuracy risk** concentrates in lowering budgets for reasoning-class requests, as seen at 1536/2048. Keep their budget unchanged.

### Gaps
- No public, training-free, single-pass difficulty predictor from prompt features has been validated on Qwen3-class models. DART needs two extra drafts per request, which would cost TTFT/energy on our SLO.
- No evidence was found on how budget choices interact with the 2% relative accuracy-loss penalty for mixed workloads like ours. Local A/B on val/final tracks is the only measurement path.
