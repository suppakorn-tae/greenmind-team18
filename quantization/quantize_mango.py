#!/usr/bin/env python3
"""
Quantize CMKL/MANGO1.5-Qwen3.5-9B with llm-compressor -> compressed-tensors checkpoint for vLLM.

  --scheme w8a8   INT8 weights+activations (GPTQ).  Recommended for A100: real INT8 tensor-core
                  GEMMs, ~14 GB, Qwen3.5-9B recipe recovers 99.4-100% on RedHat's evals.
  --scheme fp8    FP8 dynamic, data-free, ~10 min. On A100 vLLM runs it as W8A16 (Marlin).
  --scheme w4a16  INT4 weight-only (GPTQ, group 128, MSE observer). ~8 GB, fastest single-stream
                  decode, measurable accuracy loss -> only if you must shrink hardware.

Kept in BF16 on purpose: lm_head, embed_tokens (248k vocab), vision tower, Gated-DeltaNet
linear_attn projections (most quantization-sensitive part of Qwen3.5), and the MTP head
(copied back afterwards so MTP speculative decoding still works in vLLM).

Calibration (w8a8/w4a16): Thai + English chat rendered with MANGO's own chat template and
system prompt. Add your real workload with --calib-jsonl (best accuracy for your task).

Examples
  python quantize_mango.py --scheme fp8
  python quantize_mango.py --scheme w8a8
  python quantize_mango.py --scheme w8a8 --calib-jsonl /path/to/your_prompts.jsonl   # file must exist
  python quantize_mango.py --patch-aux ./out/MANGO1.5-Qwen3.5-9B-FP8                  # fix older output
"""
import argparse
import json
import os
import random
import shutil
import sys
import time

import torch
from compressed_tensors.utils import save_mtp_tensors_to_checkpoint
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from llmcompressor.utils import dispatch_for_generation
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

SYSTEM_PROMPT = (
    "You are Mango, a helpful AI assistant developed by CMKL University that specialized in "
    "Thai and English. You must respond in the language used by the user. If the user "
    "communicates in Thai, provide a culturally nuanced, polite, and grammatically correct "
    "response. If the user communicates in English, maintain high-quality, professional "
    "performance.\n\n"
    "คุณคือ Mango ผู้ช่วย AI ที่เชี่ยวชาญทั้งภาษาไทยและภาษาอังกฤษ ที่พัฒนาโดย มหาวิทยาลัยซีเอ็มเคแอล "
    "(CMKL University) โปรดตอบด้วยภาษาเดียวกับที่ผู้ใช้สื่อสารมา หากผู้ใช้ถามเป็นภาษาไทย "
    "ให้ตอบด้วยภาษาไทยที่สอดคล้องกับบริบททางวัฒนธรรม สุภาพ และถูกต้องตามหลักไวยากรณ์"
)

IGNORE = [
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*visual.*",
    "re:.*linear_attn.*",
    "re:.*mtp.*",
]


# ----------------------------------------------------------------------------- calibration
def _stream(name, split, n, seed, config=None):
    ds = load_dataset(name, config, split=split, streaming=True)
    return list(ds.shuffle(seed=seed, buffer_size=10_000).take(n))


def thai_rows(n, seed):
    out = []
    for ex in _stream("airesearch/WangchanThaiInstruct", "train", n, seed):
        user = (ex.get("Instruction") or "").strip()
        if ex.get("Input"):
            user += "\n\n" + ex["Input"].strip()
        if user and ex.get("Output"):
            out.append([{"role": "user", "content": user},
                        {"role": "assistant", "content": ex["Output"].strip()}])
    return out


def english_chat_rows(n, seed):
    out = []
    for ex in _stream("HuggingFaceH4/ultrachat_200k", "train_sft", n, seed):
        msgs = [m for m in ex["messages"] if m["role"] in ("user", "assistant")]
        if len(msgs) >= 2:
            out.append(msgs[:4])
    return out


def stem_rows(n, seed):
    out = []
    for ex in _stream("garage-bAInd/Open-Platypus", "train", n, seed):
        user = ex["instruction"] + (("\n" + ex["input"]) if ex.get("input") else "")
        out.append([{"role": "user", "content": user},
                    {"role": "assistant", "content": ex["output"]}])
    return out


def custom_rows(path, n, seed):
    """jsonl lines: {"messages":[...]} | {"prompt":..,"response":..} | {"prompt":..} | {"text":..}"""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if "messages" in ex:
                rows.append(ex["messages"])
            elif "prompt" in ex:
                m = [{"role": "user", "content": ex["prompt"]}]
                if ex.get("response"):
                    m.append({"role": "assistant", "content": ex["response"]})
                rows.append(m)
            elif "text" in ex:
                rows.append(ex["text"])
    random.Random(seed).shuffle(rows)
    return rows[:n]


def build_calibration(tok, args):
    n = args.num_samples
    plan = {"thai": 0.45, "en_chat": 0.35, "stem": 0.20}
    rows = []
    if args.calib_jsonl:
        k = int(n * args.custom_frac)
        got = custom_rows(args.calib_jsonl, k, args.seed)
        rows += got
        n -= len(got)
        print(f"[calib] custom: {len(got)}")
    fetch = {"thai": thai_rows, "en_chat": english_chat_rows, "stem": stem_rows}
    for key, frac in plan.items():
        k = max(1, round(n * frac))
        got = fetch[key](k, args.seed)
        print(f"[calib] {key}: {len(got)}")
        rows += got
    random.Random(args.seed).shuffle(rows)

    texts = []
    for r in rows:
        if isinstance(r, str):
            texts.append(r)
            continue
        msgs = r if r and r[0]["role"] == "system" else [{"role": "system", "content": SYSTEM_PROMPT}] + r
        add_gen = msgs[-1]["role"] != "assistant"
        texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_gen))

    enc = tok(texts, max_length=args.max_seq_len, truncation=True,
              padding=False, add_special_tokens=False)
    ds = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    ds = ds.filter(lambda x: len(x["input_ids"]) >= 32)
    lens = [len(x) for x in ds["input_ids"]]
    print(f"[calib] {len(ds)} samples, mean {sum(lens)/len(lens):.0f} tok, max {max(lens)}")
    return ds


# ----------------------------------------------------------------------------- output helpers
AUX_SKIP = {"config.json", "model.safetensors.index.json", "args.json", "recipe.yaml"}


def copy_aux_files(src, out):
    """Copy processor/preprocessor/chat-template files vLLM expects, without needing torchvision."""
    src_dir = src if os.path.isdir(src) else snapshot_download(
        src, allow_patterns=["*.json", "*.jinja", "*.txt", "*.model"])
    copied = []
    for f in sorted(os.listdir(src_dir)):
        s, d = os.path.join(src_dir, f), os.path.join(out, f)
        if f in AUX_SKIP or f.endswith(".safetensors") or not os.path.isfile(s) or os.path.exists(d):
            continue
        shutil.copy2(s, d)
        copied.append(f)
    print(f"[aux] copied into {out}: {copied or 'nothing missing'}")


def ensure_mtp(src, out):
    """llm-compressor's save_pretrained normally copies MTP itself; only do it if it didn't."""
    idx = os.path.join(out, "model.safetensors.index.json")
    keys = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {}
    n = sum(k.startswith("mtp") for k in keys)
    if n == 0:
        save_mtp_tensors_to_checkpoint(source_model=src, dest_dir=out)
        keys = json.load(open(idx))["weight_map"] if os.path.exists(idx) else {}
        n = sum(k.startswith("mtp") for k in keys)
    print(f"[mtp] {n} MTP tensors in checkpoint" + ("" if n else "  <-- WARNING: spec decoding won't work"))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="CMKL/MANGO1.5-Qwen3.5-9B")
    ap.add_argument("--scheme", choices=["w8a8", "fp8", "w4a16"], default="w8a8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--num-samples", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--calib-jsonl", default=None, help="your real prompts/responses (recommended)")
    ap.add_argument("--custom-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--act-offload", default="cuda:0",
                    help="where layer activations live during calibration; cuda:0 keeps the "
                         "32 GB RAM pod safe (model weights already ~19 GB in RAM)")
    ap.add_argument("--no-sample", action="store_true", help="skip post-quant test generation")
    ap.add_argument("--patch-aux", metavar="OUT_DIR",
                    help="only add missing processor/template files + MTP check to an existing output")
    args = ap.parse_args()

    if args.patch_aux:
        copy_aux_files(args.model, args.patch_aux)
        ensure_mtp(args.model, args.patch_aux)
        return

    # Fail fast, before spending minutes loading 19 GB of weights.
    if args.calib_jsonl and not os.path.isfile(args.calib_jsonl):
        sys.exit(f"[error] --calib-jsonl file not found: {os.path.abspath(args.calib_jsonl)}\n"
                 f"        Drop the flag to use the built-in Thai+English calibration set, or create the\n"
                 f"        file (one JSON per line: {{\"prompt\": \"...\", \"response\": \"...\"}}).")

    name = args.model.rstrip("/").split("/")[-1]
    out = args.out or os.path.join(os.environ.get("WORK", "."), "out", f"{name}-{args.scheme.upper()}")
    t0 = time.time()

    tok = AutoTokenizer.from_pretrained(args.model)
    ds = None if args.scheme == "fp8" else build_calibration(tok, args)  # data first: cheap to fail

    # Load on CPU; llm-compressor onloads one decoder layer at a time to the GPU.
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model, dtype="auto")

    if args.scheme == "fp8":
        recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
        oneshot(model=model, recipe=recipe)
    else:
        if args.scheme == "w8a8":
            recipe = GPTQModifier(targets="Linear", scheme="W8A8", ignore=IGNORE, dampening_frac=0.01)
        else:
            recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=IGNORE,
                                  dampening_frac=0.05, weight_observer="mse")
        oneshot(
            model=model,
            dataset=ds,
            # ds is pre-tokenized; pass the tokenizer so oneshot doesn't AutoProcessor-load
            # the Qwen3-VL processor (its video processor hard-requires torchvision).
            processor=tok,
            recipe=recipe,
            max_seq_length=args.max_seq_len,
            num_calibration_samples=len(ds),
            sequential_targets=["Qwen3_5DecoderLayer"],
            sequential_offload_device=args.act_offload,
        )
    print(f"[quant] done in {(time.time()-t0)/60:.1f} min")

    if not args.no_sample:
        dispatch_for_generation(model)
        for q in ["อธิบายความแตกต่างระหว่างฝนกับน้ำค้างให้ฟังหน่อย",
                  "A train leaves at 9:40 and arrives at 13:05. How long is the trip?"]:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": q}]
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**ids, max_new_tokens=200, do_sample=False)
            print("=" * 80, "\nQ:", q, "\nA:", tok.decode(gen[0][ids["input_ids"].shape[-1]:],
                                                       skip_special_tokens=True))

    model.save_pretrained(out, save_compressed=True)  # also copies the BF16 MTP head
    tok.save_pretrained(out)
    copy_aux_files(args.model, out)
    ensure_mtp(args.model, out)
    with open(os.path.join(out, "quantization_recipe_args.json"), "w") as f:
        json.dump(vars(args) | {"ignore": IGNORE}, f, indent=2, ensure_ascii=False)

    size = sum(os.path.getsize(os.path.join(out, x)) for x in os.listdir(out)) / 1e9
    print(f"[save] {out}  ({size:.1f} GB)  total {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()