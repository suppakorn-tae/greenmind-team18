"""Streaming request + tool-episode runner. Returns RAW measurements only (timings, token counts, finish);
SLO verdicts are computed later by gm.slo so they can be recomputed under other settings."""
import asyncio, json, math, time

import httpx

from . import config
from .accuracy import check_answer
from .trace import TOOLS, phase_of, tool_exec


def _p95(xs):
    xs = sorted(xs)
    return xs[max(0, math.ceil(0.95 * len(xs)) - 1)] if xs else None


def base_url(url):
    """Accept http://host:8000, .../ and .../v1 alike → http://host:8000."""
    url = url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def make_client():
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)   # open loop: no client-side cap
    return httpx.AsyncClient(timeout=httpx.Timeout(config.TIMEOUT_S, connect=30), limits=limits)


async def discover_model(client, base):
    resp = await client.get(base_url(base) + "/v1/models")
    resp.raise_for_status()
    return resp.json()["data"][0]["id"]


async def stream_request(client, url, model, messages, tools, max_tokens=config.MAX_TOKENS):
    payload = {"model": model, "messages": messages, "stream": True,
               "stream_options": {"include_usage": True},
               "max_tokens": max_tokens, "temperature": 0,
               "chat_template_kwargs": {"enable_thinking": True}}
    if tools:
        payload["tools"], payload["tool_choice"] = tools, "auto"
    r = {"ttft_any": None, "ttft_content": None, "span_any": None, "span_content": None, "tpot_chunk_p95": None,
         "prompt_tokens": None, "completion_tokens": None, "finish": None, "has_usage": False, "error": None,
         "e2e": None, "reasoning_chars": 0, "content": "", "tool_calls": []}
    content, calls, stamps_any, stamps_content = [], {}, [], []
    t0 = time.monotonic()

    async def consume():
        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                r["error"] = f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}"
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ch = json.loads(data)
                if ch.get("usage"):
                    r["usage"] = ch["usage"]
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    got_content = False
                    if d.get("content"):
                        content.append(d["content"]); got_content = True
                    rs = d.get("reasoning_content") or d.get("reasoning")   # never both
                    for tc in d.get("tool_calls") or []:
                        got_content = True
                        s = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "arguments": ""})
                        s["id"] = tc.get("id") or s["id"]
                        f = tc.get("function") or {}
                        s["name"] += f.get("name") or ""
                        s["arguments"] += f.get("arguments") or ""
                    if rs:
                        r["reasoning_chars"] += len(rs)
                    if rs or got_content:
                        now = time.monotonic()
                        stamps_any.append(now)
                        if got_content:
                            stamps_content.append(now)
                    if c.get("finish_reason"):
                        r["finish"] = c["finish_reason"]

    try:
        await asyncio.wait_for(consume(), config.TIMEOUT_S)
    except asyncio.TimeoutError:
        r["error"] = "timeout"
    except Exception as e:
        r["error"] = repr(e)[:300]
    r["e2e"] = time.monotonic() - t0
    r["content"] = "".join(content)
    r["tool_calls"] = [dict(v, id=v["id"] or f"call_{i}") for i, v in sorted(calls.items())]
    if stamps_any:
        r["ttft_any"], r["span_any"] = stamps_any[0] - t0, stamps_any[-1] - stamps_any[0]
    if stamps_content:
        r["ttft_content"], r["span_content"] = stamps_content[0] - t0, stamps_content[-1] - stamps_content[0]
    u = r.pop("usage", None)
    if u:
        r["has_usage"] = True
        r["prompt_tokens"], r["completion_tokens"] = u.get("prompt_tokens"), u.get("completion_tokens")
    ct = r["completion_tokens"]
    if len(stamps_any) > 1 and ct and ct > 1:   # p95 over per-token gaps (chunk gap ÷ tokens per chunk)
        per_chunk = ct / len(stamps_any)
        r["tpot_chunk_p95"] = _p95([(b - a) / per_chunk for a, b in zip(stamps_any, stamps_any[1:])])
    return r


async def run_item(client, url, model, it, t0, bounds, records, max_tokens=config.MAX_TOKENS):
    """Wait for the item's scheduled time, then run it (a tool episode = up to MAX_TURNS requests; every turn
    is its own record). → {id, cls, correct, turns, final_turn}. Appends raw records to `records`."""
    delay = t0 + it["t"] - time.monotonic()
    if delay > 0:
        await asyncio.sleep(delay)
    msgs = [dict(m) for m in it["messages"]]
    final, all_calls, turn = None, [], 0
    for turn in range(config.MAX_TURNS if it.get("episode") else 1):
        sent = time.monotonic() - t0
        r = await stream_request(client, url, model, msgs, it.get("tools"), max_tokens)
        final = r
        rec = {k: v for k, v in r.items() if k not in ("content", "tool_calls")}
        rec.update(id=it["id"], cls=it["cls"], turn=turn, sched_t=it["t"], sent=round(sent, 4),
                   launch_lag=round(sent - it["t"], 4) if turn == 0 else None, phase=phase_of(sent, bounds),
                   done=round(time.monotonic() - t0, 4), n_tool_calls=len(r["tool_calls"]),
                   answer_tail=r["content"][-200:])
        records.append(rec)
        if r["error"] or not r["tool_calls"] or r["finish"] == "length":
            break
        all_calls += r["tool_calls"]
        msgs.append({"role": "assistant", "content": r["content"] or "",
                     "tool_calls": [{"id": c["id"], "type": "function",
                                     "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
                                    for c in r["tool_calls"]]})
        for c in r["tool_calls"]:
            msgs.append({"role": "tool", "tool_call_id": c["id"], "content": tool_exec(c["name"], c["arguments"])})
    correct = bool(final) and check_answer(it["check"], final["content"], all_calls)
    return {"id": it["id"], "cls": it["cls"], "correct": correct, "turns": turn + 1, "final_turn": turn}


async def loop_lag_monitor(stats, interval=0.05):
    """Event-loop lag: how late a sleep(interval) wakes. Keeps the max in stats['max_loop_lag_s']."""
    while True:
        t = time.monotonic()
        await asyncio.sleep(interval)
        stats["max_loop_lag_s"] = max(stats.get("max_loop_lag_s", 0.0), time.monotonic() - t - interval)


async def health(client, base, model=None):
    """Mirror of the grader's health check → (model, [(name, ok, detail)])."""
    base = base_url(base)
    url = base + "/v1/chat/completions"
    checks = []
    model = model or await discover_model(client, base)
    r = await stream_request(client, url, model, [{"role": "user", "content": "What is 17*23? Answer briefly."}], None, 2048)
    good = r["ttft_any"] is not None and bool(r["completion_tokens"]) and not r["error"] and r["has_usage"]
    checks.append(("stream+usage", good, f"ttft={r['ttft_any']} tokens={r['completion_tokens']} "
                   f"reasoning_chars={r['reasoning_chars']} finish={r['finish']} err={r['error']}"))
    r = await stream_request(client, url, model, [{"role": "user", "content": "What's the weather in Chiang Mai right now?"}], TOOLS, 2048)
    tc = r["tool_calls"]
    good = bool(tc) and tc[0]["name"] == "get_weather" and "chiang" in tc[0]["arguments"].lower()
    checks.append(("tool calling", good, f"calls={tc} finish={r['finish']} err={r['error']}"))
    return model, checks
