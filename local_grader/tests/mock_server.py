"""OpenAI-compatible SSE mock server for local LLM-grader tests.

Tiny "model": answers are derived from the prompt text and the tool messages of
the conversation (it never calls the real generators, it just behaves like one).
Serves GET /v1/models, POST /v1/chat/completions (SSE) and GET /stats.
"""
from __future__ import annotations

import asyncio
import json
import re
import socket
import threading
import time
from dataclasses import dataclass

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

WAREHOUSES = ["BKK-1", "CNX-2", "HKT-3"]

THAI_FACTS = (
    ("วันสงกรานต์ตรงกับวันที่เท่าใดของเดือนเมษายน", "13"),
    ("ดอยอินทนนท์", "เชียงใหม่"),
    ("แม่น้ำสายหลัก", "เจ้าพระยา"),
    ("สกุลเงิน", "บาท"),
    ("เกาะที่ใหญ่ที่สุด", "ภูเก็ต"),
    ("อุทยานแห่งชาติแห่งแรก", "เขาใหญ่"),
    ("ราชธานี", "ธนบุรี"),
)


@dataclass
class MockConfig:
    token_interval: float = 0.01
    ttft_delay: float = 0.05
    n_reasoning_tokens: int = 20
    fail_every: int = 0


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _last_user_text(messages) -> str:
    for m in reversed(messages or []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                c = json.dumps(c)
            return c or ""
    return ""


def _pending_tool_messages(messages):
    """Tool messages that come after the last user message (pending tool results)."""
    last_user = -1
    for i, m in enumerate(messages or []):
        if m.get("role") == "user":
            last_user = i
    return [m for m in (messages or [])[last_user + 1:] if m.get("role") == "tool"]


def _results_of(tool_msgs):
    out = []
    for m in tool_msgs or []:
        try:
            out.append(json.loads(m.get("content") or "{}"))
        except Exception:
            out.append({})
    return out


def _fmt(x) -> str:
    x = float(x)
    return str(int(x)) if x == int(x) else f"{x:.2f}"


def _boxed(x) -> str:
    return "\\boxed{" + _fmt(x) + "}"


def _final_answer(user_text: str) -> str:
    """Answer a non-tool prompt straight from its text."""
    text = user_text or ""
    codes = re.findall(r"The access code for vault ([A-Z0-9]+) is ([A-Z0-9\-_]+)\.", text)
    if codes:
        asked = re.search(r"vault ([A-Z0-9]+)", (text.split("Question:")[-1] if "Question:" in text else text))
        for v, c in codes:
            if not asked or v == asked.group(1):
                return c
        return codes[0][1]
    if re.search(r"def f\(xs\)|for xs =", text):
        xs = re.search(r"xs = (\[[^\]]*\])", text)
        vals = [float(v) for v in re.findall(r"-?\d+", xs.group(1))] if xs else []
        return f"The function returns {_boxed(sum(v * v for v in vals if v % 3 == 0))}."
    m = re.search(r"pencils at (\d+) baht each and notebooks at (\d+) baht each", text)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        mx = re.search(r"Mali buys (\d+) pencils and (\d+) notebooks", text)
        x, y = float(mx.group(1)), float(mx.group(2))
        z = float(re.search(r"pays with a (\d+) baht note", text).group(1))
        return f"Mali's change is {_boxed(z - a * x - b * y)} baht."
    m = re.search(r"What is (\d+)\^(\d+) mod (\d+)\?", text)
    if m:
        return f"The answer is {_boxed(pow(int(m.group(1)), int(m.group(2)), int(m.group(3))))}."
    if re.search(r"prime numbers less than", text):
        n = int(re.search(r"less than (\d+)", text).group(1))
        ans = sum(p for p in range(2, n) if all(p % d for d in range(2, int(p ** 0.5) + 1)))
        return f"The sum of primes below {n} is {_boxed(ans)}."
    for marker, word in THAI_FACTS:
        if marker in text:
            return word
    if "ธนบัตร" in text:  # Thai change problem: <a> baht/kg + <b> baht/kg, buys <x> + <y>
        nums = [int(v) for v in re.findall(r"\d+", text)]
        if len(nums) >= 5:
            a, b, x, y, card = nums[0], nums[1], nums[2], nums[3], nums[4]
            return f"Somchai's change is {_boxed(card - a * x - b * y)} baht."
    return "Here is a helpful answer to your question."


def _price_of(results, sku: str) -> float:
    want = (sku or "").upper()
    for r in results or []:
        if str(r.get("sku") or "").upper() == want and r.get("unit_price_thb") is not None:
            return float(r["unit_price_thb"])
    return 0.0


def _plan(messages, tools):
    """→ (tool_calls, final_text); tool_calls == None means "answer directly".

    tool_calls is a list of {"name": ..., "arguments": {...}} for this turn.
    """
    user_text = _last_user_text(messages)
    results = _results_of(_pending_tool_messages(messages))
    low = (user_text or "").lower()

    if tools:
        if re.search(r"total cost|current unit prices", low):
            pairs = re.findall(r"(\d+)\s+x\s+(SKU-\d+)", user_text) or [
                (1, s) for s in re.findall(r"SKU-\d+", user_text)
            ]
            if results:
                total = sum(int(q) * _price_of(results, s) for q, s in pairs)
                return None, f"The grand total cost is {_boxed(total)} THB."
            return [{"name": "get_product_price", "arguments": {"sku": s}} for _, s in pairs], None
        if re.search(r"in stock in total", low):
            if results:
                total = sum(int(r.get("units_in_stock") or 0) for r in results)
                return None, f"The combined stock across those warehouses is {_boxed(total)}."
            sku = (re.search(r"of (SKU-\d+)", user_text) or [None, "SKU-0000"])[1]
            return [{"name": "get_stock", "arguments": {"sku": sku, "warehouse": w}} for w in WAREHOUSES], None
        if "unit price" in low and "convert" in low:
            if results and "result" in results[-1]:
                return None, f"The converted amount is {_boxed(results[-1].get('result'))}."
            if results and "unit_price_thb" in results[-1]:
                cur = (re.search(r"in (USD|EUR|JPY|THB)", user_text) or [None, "USD"])[1]
                return [{"name": "convert_currency",
                         "arguments": {"amount": float(results[-1].get("unit_price_thb") or 0),
                                        "from_currency": "THB", "to_currency": cur}}], None
            sku = (re.search(r"of (SKU-\d+)", user_text) or [None, "SKU-0000"])[1]
            return [{"name": "get_product_price", "arguments": {"sku": sku}}], None
        if "weather" in low:
            city = (re.search(r"weather (?:like )?in (.+?) right now", user_text) or [None, "Bangkok"])[1]
            city = city.strip().strip("?").strip()
            if results and "temp_c" in results[0]:
                return None, (f"The weather in {results[0].get('city') or city} is "
                              f"{results[0].get('temp_c')}C and {results[0].get('condition')}. "
                              f"You should probably bring an umbrella.")
            return [{"name": "get_weather", "arguments": {"city": city.lower()}}], None
        return None, "I am not able to help with that yet."

    return None, _final_answer(user_text)


def _build(cfg: MockConfig, messages, tool_calls, final_text):
    """Return (token_chunks, finish_reason, usage) for the SSE stream."""
    token_chunks = []
    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
        token_chunks = [{"choices": [{"index": 0, "delta": {"reasoning_content": f" tok{i}"}}]} for i in range(cfg.n_reasoning_tokens)]
        for i, call in enumerate(tool_calls):
            token_chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [
                {"index": i, "id": f"call_{i}", "type": "function",
                 "function": {"name": call["name"], "arguments": ""}}]}}]})
            args = json.dumps(call["arguments"])
            mid = len(args) // 2
            parts = [args[:mid], args[mid:]] if len(args) > 1 else [args]
            for p in parts:
                token_chunks.append({"choices": [{"index": 0, "delta": {"tool_calls": [
                    {"index": i, "function": {"arguments": p}}]}}]})
    else:
        content = final_text or ""
        toks = re.findall(r"\S+", content) or ["Here is a helpful answer to your question."]
        token_chunks = [{"choices": [{"index": 0, "delta": {"reasoning_content": f" tok{i}"}}]} for i in range(cfg.n_reasoning_tokens)]
        token_chunks += [{"choices": [{"index": 0, "delta": {"content": t}}]} for t in toks]
    p_tokens = len(json.dumps(messages, ensure_ascii=False)) // 4
    usage = {"prompt_tokens": p_tokens, "completion_tokens": len(token_chunks), "total_tokens": p_tokens + len(token_chunks)}
    return token_chunks, finish_reason, usage


async def _events(cfg: MockConfig, messages, tool_calls, final_text):
    token_chunks, finish_reason, usage = _build(cfg, messages, tool_calls, final_text)
    await asyncio.sleep(cfg.ttft_delay)
    for i, ch in enumerate(token_chunks):
        yield f"data: {json.dumps(ch)}\n\n"
        if i < len(token_chunks) - 1:
            await asyncio.sleep(cfg.token_interval)
    yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish_reason}]})}\n\n"
    yield f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n"
    yield "data: [DONE]\n\n"


def create_app(cfg: MockConfig | None = None) -> Starlette:
    cfg = cfg or MockConfig()
    state: dict = {"requests": 0, "inflight": 0, "max_concurrent": 0}

    async def models(_: Request) -> JSONResponse:
        return JSONResponse({"data": [{"id": "mock-model"}]})

    async def stats(_: Request) -> JSONResponse:
        return JSONResponse({"requests": state["requests"], "max_concurrent": state["max_concurrent"]})

    async def chat(request: Request) -> Response:
        body = await request.json()
        state["requests"] += 1
        state["inflight"] += 1
        state["max_concurrent"] = max(state["max_concurrent"], state["inflight"])
        try:
            if cfg.fail_every > 0 and state["requests"] % cfg.fail_every == 0:
                return JSONResponse(status_code=500, content={"error": "mock server failure"})
            messages = body.get("messages") or []
            tools = body.get("tools")
            tool_calls, final_text = _plan(messages, tools)
            if body.get("stream", True):
                return StreamingResponse(_events(cfg, messages, tool_calls, final_text), media_type="text/event-stream")
            token_chunks, finish_reason, usage = _build(cfg, messages, tool_calls, final_text)
            return JSONResponse({
                "choices": [{"index": 0, "message": {"role": "assistant", "content": final_text or ""},
                             "finish_reason": finish_reason}],
                "usage": usage,
            })
        finally:
            state["inflight"] -= 1

    return Starlette(routes=[
        Route("/v1/models", models, methods=["GET"]),
        Route("/stats", stats, methods=["GET"]),
        Route("/v1/chat/completions", chat, methods=["POST"]),
    ])


class MockServer:
    def __init__(self, cfg: MockConfig | None = None):
        self.cfg = cfg or MockConfig()
        self.app = create_app(self.cfg)
        self._server = None
        self._thread = None
        self.port = _free_port()

    def start(self, port: int = 0) -> "MockServer":
        if port:
            self.port = port
        server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="warning"))
        self._server = server
        self._thread = threading.Thread(target=server.run, daemon=True)
        self._thread.start()
        deadline = time.time() + 20
        while not server.started:
            if time.time() > deadline:
                raise RuntimeError("mock server failed to start")
            time.sleep(0.005)
        return self

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def __enter__(self) -> "MockServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
