---
base_model: CMKL/MANGO1.5-Qwen3.5-9B
base_model_relation: quantized
language:
  - th
  - en
license: apache-2.0
license_link: https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE
tags:
  - w8a8
  - int8
  - compressed-tensors
  - llm-compressor
  - vllm
  - thai
  - bilingual
  - low-resource
  - lora
  - supervised-fine-tuning
  - qwen3.5
pipeline_tag: text-generation
library_name: transformers
---

# MANGO1.5-Qwen3.5-9B-W8A8

INT8 weight + INT8 dynamic activation (W8A8) quantization of
[`CMKL/MANGO1.5-Qwen3.5-9B`](https://huggingface.co/CMKL/MANGO1.5-Qwen3.5-9B), made with
llm-compressor (GPTQ, block 128, static actorder) and saved in `compressed-tensors` format for vLLM.
Calibrated on a Thai+English mix (WangchanThaiInstruct, UltraChat, Open-Platypus).
`lm_head`, embeddings, vision, linear-attention and MTP layers are kept in BF16; the MTP head ships
as `model_mtp.safetensors` for speculative decoding. See `recipe.yaml`.

```bash
vllm serve suppakorn-k/MANGO1.5-Qwen3.5-9B-W8A8
```

The original model card follows.

---

# MANGO1.5-Qwen3.5-9B

**Thai MANGO** (Thai Multi-modal Adaptive Neural Generative Orchestrator) is CMKL University's Thai-language fine-tuning effort. **MANGO1.5-Qwen3.5-9B** is the newest release in the line, a dense 9B text-only bilingual (Thai/English) fine-tune of [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B), produced via full-model LoRA supervised fine-tuning on a 5,000,000-sample English/Thai instruction corpus.

It supersedes the previous release, [`CMKL/MANGO-Qwen3-Omni-30B-A3B-Instruct`](https://huggingface.co/CMKL/MANGO-Qwen3-Omni-30B-A3B-Instruct) (a 30B MoE omni-modal fine-tune), with a different design point: a smaller, dense, text-only base trained on a substantially larger and more broadly-sourced bilingual SFT corpus (5M vs. ~108K samples). This checkpoint is **stage 1 (SFT)** of a two-stage post-training plan; a preference-optimization stage (DPO) targeting Thai response quality specifically is planned as a follow-up and is not part of this release.

## Model Details

- **Base model:** [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) — a dense 9B-parameter model from the Qwen3.5 family.
- **Fine-tuning method:** LoRA (rank 64, alpha 128, dropout 0.05, `target_modules=all-linear`), applied across the full language model (no frozen sub-modules). Weights in this repository are the **LoRA adapter merged into the base weights** and exported to standard HF safetensors — this is a complete, standalone checkpoint, not an adapter to be applied separately.
- **Training data:** 5,000,000 samples (65% English / 35% Thai), text-only instruction-tuning data spanning chat, math, code, STEM, knowledge, creative writing, summarization, translation, safety, legal, medical, and exam domains. See **Training Data** below for the full domain/language composition; source-level dataset provenance is not published with this release.
- **Modality:** text input, text output. The base model's architecture (`Qwen3_5ForConditionalGeneration`) supports multimodal input per its own model card, but this fine-tune was performed on a text-only corpus — no vision data was used and no vision-related behavior was targeted, evaluated, or should be assumed improved here.
- **Languages:** Thai, English (mixed/code-switched input handled by the same filtering discipline used to build the training data — see below).
- **License:** inherits the Apache 2.0 license of the base model. See [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B/blob/main/LICENSE) for exact terms.

## Intended Use

This model is intended for:
- Thai-language and English-language chat, instruction-following, reasoning, summarization, translation, and Q&A.
- Bilingual use cases where a single model needs to respond fluently in whichever of the two languages the user writes in.

It is trained with a default system prompt that establishes this bilingual persona (see **How to Use**). It is a research/engineering artifact from CMKL University, not a production-hardened or safety-certified assistant — it inherits the base model's general limitations (hallucination, non-factual output, sensitivity to prompt phrasing), and has only the safety-domain SFT data folded in at this stage (no dedicated safety RLHF pass yet).

## How to Use

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "CMKL/MANGO1.5-Qwen3.5-9B"
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

system_prompt = (
    "You are Mango, a helpful AI assistant developed by CMKL University that specialized in "
    "Thai and English. You must respond in the language used by the user. If the user "
    "communicates in Thai, provide a culturally nuanced, polite, and grammatically correct "
    "response. If the user communicates in English, maintain high-quality, professional "
    "performance.\n\n"
    "คุณคือ Mango ผู้ช่วย AI ที่เชี่ยวชาญทั้งภาษาไทยและภาษาอังกฤษ ที่พัฒนาโดย มหาวิทยาลัยซีเอ็มเคแอล "
    "(CMKL University) โปรดตอบด้วยภาษาเดียวกับที่ผู้ใช้สื่อสารมา หากผู้ใช้ถามเป็นภาษาไทย "
    "ให้ตอบด้วยภาษาไทยที่สอดคล้องกับบริบททางวัฒนธรรม สุภาพ และถูกต้องตามหลักไวยากรณ์"
)

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "อธิบายความแตกต่างระหว่างฝนกับน้ำค้างให้ฟังหน่อย"},
]

inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
output = model.generate(inputs, max_new_tokens=512)
print(tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True))
```

This system prompt is only a fallback default used during training — rows that already carried their own system message in the training data kept it, so you're free to supply a different system prompt at inference time.

### Serving with vLLM

[vLLM](https://github.com/vllm-project/vllm) is the recommended way to serve this checkpoint — it's the runtime this model was validated on, including the speculative-decoding path below. Requires a vLLM build with Qwen3.5 hybrid-attention (Gated DeltaNet) support (validated on 0.24.0–0.27.1).

```bash
vllm serve CMKL/MANGO1.5-Qwen3.5-9B \
  --port 8000 --tensor-parallel-size 1 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

This checkpoint includes the base model's Multi-Token-Prediction (MTP/NextN) draft layer (see **Model Details**), so self-speculative decoding can be enabled directly, with no separate draft model:

```bash
vllm serve CMKL/MANGO1.5-Qwen3.5-9B \
  --port 8000 --tensor-parallel-size 1 --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --speculative-config '{"model": "CMKL/MANGO1.5-Qwen3.5-9B", "num_speculative_tokens": 3, "method": "mtp"}'
```

We've validated `"method": "mtp"` end-to-end (64–76% draft-token acceptance in our own smoke tests); the upstream [Qwen3.5-9B base model card](https://huggingface.co/Qwen/Qwen3.5-9B) documents `"method": "qwen3_next_mtp"` instead — the accepted method string has moved between vLLM versions, so try `mtp` first and fall back to `qwen3_next_mtp` if your vLLM build rejects it.

### Serving with SGLang

[SGLang](https://github.com/sgl-project/sglang) also supports Qwen3.5's architecture (per the [base model card](https://huggingface.co/Qwen/Qwen3.5-9B)); we have not personally validated SGLang against this fine-tune (only vLLM, above), but the same recipe should apply directly since the architecture and weights are unchanged from base:

```bash
python -m sglang.launch_server --model-path CMKL/MANGO1.5-Qwen3.5-9B \
  --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 262144 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

With MTP speculative decoding:

```bash
python -m sglang.launch_server --model-path CMKL/MANGO1.5-Qwen3.5-9B \
  --port 8000 --tp-size 1 --mem-fraction-static 0.8 --context-length 262144 \
  --reasoning-parser qwen3 \
  --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
```

## Training Data

5,000,000 samples total, English/Thai bilingual, text-only instruction-tuning data. Composition:

### Language ratio

| Language | Samples | % of total |
|---|---:|---:|
| English | 3,250,000 | 65.0% |
| Thai | 1,750,000 | 35.0% |

### Domain ratio (% of full 5,000,000)

| Domain | % of total |
|---|---:|
| chat | 27.50% |
| math | 12.04% |
| code | 9.76% |
| summarization | 9.84% |
| stem | 9.68% |
| creative | 6.06% |
| knowledge | 5.53% |
| translation | 5.13% |
| instruction_following | 4.15% |
| safety | 4.11% |
| medical | 2.66% |
| legal | 1.25% |
| finance | 0.69% |
| tool_calling | 0.59% |
| wiki_qa | 0.87% |
| customer_service | 0.10% |
| exam | 0.04% |

Some domains are English-only (STEM, knowledge, finance, instruction-following, tool-calling) or Thai-only (translation, legal, medical, wiki QA, customer service, exam) by construction — the source pools for those domains only existed in one of the two languages at adequate quality/volume.

### Data preparation

- **Language-purity filtering:** every row was scored for script composition (Thai-Unicode, Latin, other) and language-ID confidence before being allocated to its language bucket. Rows dominated by a third script (CJK, Cyrillic, Arabic, etc.) were rejected outright. Genuine bilingual/translation pairs are deliberately preserved in the Thai bucket rather than treated as contamination.
- **Deduplication:** global exact-duplicate removal via content hashing before allocation.
- **Domain-balanced allocation:** a proportional water-filling allocation targets a fixed domain mix per language without exceeding any domain's real post-filter data pool, and caps how much of a single domain's final allocation any one source can supply, to avoid one large or templated source dominating a domain.
- **Benchmark-contamination audit:** every candidate source was checked for `test`/`dev`/`validation` splits before inclusion, and known public benchmark sets were excluded from training entirely.

Source-level dataset provenance (which repositories contributed which rows) is intentionally not disclosed with this release.

## Training Details

| | |
|---|---|
| Framework | [ms-swift](https://github.com/modelscope/ms-swift) v4.4.2, Megatron-SWIFT backend (Megatron-core 0.18.2), mcore-bridge for direct HF-format weight loading/export |
| Hardware | 1 node × 8× NVIDIA A100 40GB |
| Parallelism | Tensor parallel = 4, Data parallel = 2, Pipeline parallel = 1, sequence parallel enabled |
| LoRA config | rank 64, alpha 128, dropout 0.05, `target_modules=all-linear`, full language model (no frozen sub-modules) |
| Optimizer | Megatron Adam, LR 1e-4, 5% linear warmup, min LR 1e-5, weight decay 0.1, betas (0.9, 0.95) |
| Batch size | global 128, micro-batch 4/device, sequence packing enabled |
| Sequence length | max 8,192 tokens (packed; longer samples dropped, not truncated) |
| Epochs | 3 |
| Precision / kernels | bf16 compute, Flash Attention, full uniform activation recomputation |
| Wall-clock | ~9 days on the above hardware |
| Final eval loss | 0.685 (held-out split, end of epoch 3) |

## Evaluation

Evaluated on the [ThaiLLM Leaderboard](https://opentyphoon.ai/blog/en/introducing-the-thaillm-leaderboard-thaillm-evaluation-ecosystem-508e789d06bf) methodology — 9 datasets across three categories — against its own base model ([Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)) and against [Typhoon2.1-Gemma3-12B](https://huggingface.co/scb10x/typhoon2.1-gemma3-12b), a well-established 12B Thai-LLM effort from SCB10X. All rows were scored by the same harness with identical prompts per dataset; the Mango/base rows share an identical serving setup (vLLM, bf16, greedy decoding, 16,384-token context budget), while Typhoon2.1-Gemma3-12B was served with its own repo's recommended chat template and default decoding settings, so minor serving differences there are possible. The fine-tune was evaluated with reasoning toggled on and off, since Qwen3.5's hybrid-thinking template supports both.

**Mango1.5 (think mode) has the highest normalized average (0.444)** of the four — ahead of Typhoon2.1-Gemma3-12B (0.349) despite being a smaller 9B model, and far ahead of its own un-tuned base (0.305). Typhoon2.1-Gemma3-12B does still lead on two individual datasets: Wisesight sentiment (47.0% vs. Mango's 45.5%) and Flores200 BLEU (38.3 vs. 37.0).

| Category | Dataset | Mango (think) | Mango (no-think) | Qwen3.5-9B base¹ | Typhoon2.1-Gemma3-12B |
|---|---|---:|---:|---:|---:|
| Exam | ThaiExam (accuracy) | **60.0%** | 35.2% | 30.0% | 44.5% |
| Exam | M3Exam-Thai (accuracy) | **65.4%** | 55.4% | 21.6% | 18.4% |
| NLU | Belebele (accuracy) | **80.0%** | 72.1% | 58.1% | 62.8% |
| NLU | XNLI (accuracy) | 54.7% | **57.7%** | 43.7% | 50.4% |
| NLU | XCOPA (accuracy) | **86.6%** | 73.0% | 53.8% | 67.4% |
| NLU | Wisesight sentiment (accuracy) | 45.5% | 37.6% | 39.0% | **47.0%** |
| NLG | XLSum (ROUGE-L) | **0.0452** | 0.0260 | 0.0277 | 0.0448 |
| NLG | Flores200 (BLEU, en↔th avg) | 34.1 | 37.0 | 33.3 | **38.3** |
| NLG | iApp Wiki QA (F1) | 67.6% | **69.0%** | 51.2% | 50.0% |
| **Overall** | normalized average² | **0.444** | 0.405 | 0.305 | 0.349 |

The fine-tune outperforms its own base model on every dataset in both reasoning modes. Reasoning drives the largest gains on the Exam category specifically (ThaiExam +24.8pp, M3Exam +9.9pp think vs. no-think) with comparatively little effect on NLG quality — consistent with chain-of-thought helping multiple-choice/classification tasks more than open-ended generation.

¹ *Qwen3.5-9B base was evaluated with reasoning forced off (`enable_thinking: false`). Left at its own default (reasoning on), it deliberates far past what's practical for a full run — e.g. looping through repeated draft translations of a single sentence and hitting a 2,048-token generation cap still undecided, where this fine-tune closes the same task in under 750 tokens. That gap — reasoning efficiently vs. not at all — is itself an effect of the fine-tuning.*

² *Mean of accuracy (Exam, NLU) and ROUGE-L / (BLEU÷100) / F1 (NLG). MT-Bench (LLM-as-judge) was not run — no judge model was configured for this evaluation pass.*

## Limitations

- Evaluation to date covers its own base model plus one other organization's model (Typhoon2.1-Gemma3-12B; see **Evaluation**) — no broader sweep across other Thai LLMs yet, and no LLM-as-judge (MT-Bench) pass.
- LoRA fine-tuning at this data scale (5M samples, ~5–10B+ tokens) may underfit relative to full fine-tuning; this is a known open question for the chosen rank and has not been directly measured against a full-FT counterpart.
- No dedicated preference-alignment (DPO/RLHF) pass yet — this is an SFT-only checkpoint. Response style, refusal calibration, and Thai-specific fluency/register are expected to improve in a planned follow-up stage.
- Domain coverage is uneven by language (see **Training Data**) — several domains exist in only one of the two languages, so cross-lingual transfer for those domains is untested.
- Not intended for safety-critical deployment without additional evaluation.

## Contributor

- Atikan Payungthong (atikan@cmkl.ac.th)
- Dr. Akkarit Sangpetch (akkarit@cmkl.ac.th)

## Related Artifacts

- Previous release: [CMKL/MANGO-Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/CMKL/MANGO-Qwen3-Omni-30B-A3B-Instruct)
- Base model: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
