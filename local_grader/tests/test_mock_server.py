import asyncio
import json

import httpx
import pytest

import gm.accuracy as accuracy
import gm.trace as trace
from mock_server import MockConfig, MockServer

MODEL = "mock-model"
CLASSES = ("agent", "rag", "long", "chat", "reason", "thai")


def _parse_sse(text):
    events, done = [], False
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            done = True
            break
        events.append(json.loads(payload))
    return events, done


def _token_events(events):
    tokens, finish, usage = [], None, None
    for e in events:
        choices = e.get("choices") or []
        if not choices:
            usage = e.get("usage")
            continue
        ch = choices[0]
        if ch.get("finish_reason"):
            finish = ch
            continue
        tokens.append(e)
    return tokens, finish, usage


def test_sse_shape(mock_server):
    r = httpx.post(f"{mock_server.url}/v1/chat/completions",
                   json={"model": MODEL, "messages": [{"role": "user", "content": "Hello, how are you?"}],
                         "stream": True}, timeout=30.0)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events, done = _parse_sse(r.text)
    assert done
    tokens, finish, usage = _token_events(events)
    assert finish is not None and finish["finish_reason"] == "stop"
    assert events[-1].get("choices") == [] and usage
    n = mock_server.cfg.n_reasoning_tokens
    assert len(tokens) >= n
    for e in tokens[:n]:
        assert "reasoning_content" in (e["choices"][0].get("delta") or {})
    for e in tokens[n:]:
        assert "content" in (e["choices"][0].get("delta") or {})
    assert usage["completion_tokens"] == len(tokens)
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["prompt_tokens"] == len(json.dumps([{"role": "user", "content": "Hello, how are you?"}])) // 4


def test_models_and_stats(mock_server):
    r = httpx.get(f"{mock_server.url}/v1/models", timeout=30.0)
    assert r.status_code == 200 and r.json() == {"data": [{"id": "mock-model"}]}
    s = httpx.get(f"{mock_server.url}/stats", timeout=30.0)
    assert s.status_code == 200 and set(s.json()) == {"requests", "max_concurrent"}


async def _run_episode(client, item):
    """Replay one trace item (optionally multi-turn) against the mock; → (final_text, calls)."""
    messages = [dict(m) for m in item.get("messages") or []]
    tools = item.get("tools")
    calls: list = []
    final_text = ""
    for _ in range(15):
        payload = {"model": MODEL, "messages": messages, "stream": True}
        if tools is not None:
            payload["tools"] = tools
        content = ""
        acc: dict = {}
        finish_reason = None
        async with client.stream("POST", "/v1/chat/completions", json=payload) as r:
            assert r.status_code == 200, f"item {item.get('id')} got HTTP {r.status_code}"
            async for line in r.aiter_lines():
                line = (line or "").strip()
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                choices = ev.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    content += delta["content"]
                for tc in delta.get("tool_calls") or []:
                    idx = tc.get("index") or 0
                    fn = tc.get("function") or {}
                    if tc.get("id"):
                        acc[idx] = {"id": tc["id"], "name": fn.get("name"), "arguments": fn.get("arguments") or ""}
                    else:
                        acc[idx]["arguments"] += fn.get("arguments") or ""
        final_text = content
        if finish_reason == "tool_calls" and acc:
            calls_list = [acc[k] for k in sorted(acc)]
            calls.extend(calls_list)
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": [{"id": c["id"], "type": "function",
                                "function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls_list],
            })
            for c in calls_list:
                messages.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                                 "content": trace.tool_exec(c["name"], c["arguments"])})
            continue
        break
    return final_text, calls


def test_full_trace():
    with MockServer(MockConfig(token_interval=0.002, ttft_delay=0.002)) as srv:
        items, _bounds = trace.build_trace(100, 0)

        async def main():
            async with httpx.AsyncClient(base_url=srv.url, timeout=30.0) as client:
                pairs = await asyncio.gather(*[_run_episode(client, it) for it in items])
            return pairs

        pairs = asyncio.run(main())

        by_cls: dict = {}
        for it, (final_text, calls) in zip(items, pairs):
            ok = accuracy.check_answer(it["check"], final_text, calls or [])
            by_cls.setdefault(it["cls"], []).append((it, ok))
        for cls in CLASSES:
            entries = by_cls.get(cls, [])
            acc = sum(1 for _it, ok in entries if ok) / len(entries) if entries else 0.0
            assert acc == 1.0, f"class {cls}: accuracy {acc}, n={len(entries)}"


def test_fail_every():
    with MockServer(MockConfig(fail_every=3)) as srv:
        codes = [httpx.post(f"{srv.url}/v1/chat/completions",
                            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                                  "stream": True}, timeout=30.0).status_code for _ in range(3)]
    assert codes == [200, 200, 500]
