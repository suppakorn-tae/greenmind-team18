#!/usr/bin/env bash
# A/B accuracy + speed check against a RUNNING vLLM server (serve.sh). Run once per checkpoint:
#   bash serve.sh CMKL/MANGO1.5-Qwen3.5-9B            &  bash eval_ab.sh bf16  CMKL/MANGO1.5-Qwen3.5-9B
#   bash serve.sh ~/mango-quant/out/...-W8A8          &  bash eval_ab.sh w8a8  ~/mango-quant/out/...-W8A8
# Then: python compare.py results/*   (or read the printed tables)
# Tasks mirror MANGO's ThaiLLM eval (Belebele, XCOPA, XNLI) + Thai MGSM; loglik tasks are
# very sensitive to quantization noise, so they catch damage cheaply. Also run your own grader.
set -euo pipefail
LABEL="${1:?label e.g. bf16|w8a8}"; TOKENIZER="${2:?model path or HF id (for tokenizer)}"
PORT="${PORT:-8000}"; OUT="${OUT:-results}/$LABEL"; LIMIT="${LIMIT:-}"   # LIMIT=200 for a quick pass
mkdir -p "$OUT"

until curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do sleep 5; done

lm_eval --model local-completions \
  --model_args "model=mango,base_url=http://localhost:$PORT/v1/completions,tokenizer=$TOKENIZER,num_concurrent=32,max_retries=3" \
  --tasks belebele_tha_Thai,xcopa_th,xnli_th \
  ${LIMIT:+--limit $LIMIT} --output_path "$OUT/loglik" 2>&1 | tee "$OUT/loglik.log" | tail -15

lm_eval --model local-chat-completions \
  --model_args "model=mango,base_url=http://localhost:$PORT/v1/chat/completions,num_concurrent=32,max_retries=3,tokenized_requests=False" \
  --apply_chat_template --tasks mgsm_native_cot_th \
  --gen_kwargs "temperature=0,max_gen_toks=4096" \
  ${LIMIT:+--limit $LIMIT} --output_path "$OUT/mgsm" 2>&1 | tee "$OUT/mgsm.log" | tail -10

# throughput / latency at your realistic concurrency
vllm bench serve --backend openai-chat --base-url "http://localhost:$PORT" \
  --endpoint /v1/chat/completions --model mango --tokenizer "$TOKENIZER" \
  --dataset-name random --random-input-len 1024 --random-output-len 512 \
  --num-prompts 256 --max-concurrency "${CONC:-32}" 2>&1 | tee "$OUT/bench.log" | tail -25
