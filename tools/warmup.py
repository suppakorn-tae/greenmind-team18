#!/usr/bin/env python3
"""Pre-measurement server warmup: triggers Triton/CUDA-graph JIT for batch sizes, prompt lengths, spec-decode,
tool-call paths. Uses RANDOM prompts (no overlap with test data, no response caching involved). Run once after start."""
import asyncio, json, random, sys, httpx
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
rnd = random.Random(12345)
WORDS = "alpha river stone market signal garden thread engine paper window orbit cotton harbor lantern pixel velvet".split()
def prompt(n_words): return " ".join(rnd.choice(WORDS) + str(rnd.randint(0, 999)) for _ in range(n_words))
TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a value",
          "parameters": {"type": "object", "properties": {"key": {"type": "string"}}, "required": ["key"]}}}]
async def one(c, words, max_tokens, tools=False):
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, "temperature": 0,
            "max_tokens": max_tokens, "messages": [{"role": "user", "content": "Summarize or continue: " + prompt(words)}]}
    if tools:
        body["tools"] = TOOLS; body["tool_choice"] = "auto"
        body["messages"] = [{"role": "user", "content": "Use the lookup tool for key " + prompt(3)}]
    async with c.stream("POST", URL + "/v1/chat/completions", json=body, timeout=300) as r:
        async for _ in r.aiter_lines(): pass
async def main():
    global MODEL
    async with httpx.AsyncClient() as c:
        MODEL = (await c.get(URL + "/v1/models")).json()["data"][0]["id"]
        # ramp concurrency 1..64 so every CUDA-graph / kernel batch shape is touched, mixed prompt lengths
        for conc in (1, 2, 4, 8, 16, 32, 64):
            await asyncio.gather(*[one(c, rnd.choice([30, 300, 1500, 4000, 9000]), 160, tools=(i % 5 == 0)) for i in range(conc)])
        await asyncio.gather(*[one(c, 12000, 64) for _ in range(3)])  # long chunked prefill
    print("warmup done")
asyncio.run(main())
