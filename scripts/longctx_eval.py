#!/usr/bin/env python3
"""Synthetic long-context retrieval A/B test (RULER-style key-value retrieval), generated locally: no test items.

Each prompt is a long "registry" of near-identical records (`Room <code>: access PIN is <pin>.`) mixed with filler
prose, ~LEN tokens. The question asks for the PINs of 2 rooms; correct = both PINs in the final answer. Requests are
sent like the grader's (chat completions, streaming, temperature 0, max_tokens 8192, thinking on), CONC at a time.

Usage: scripts/longctx_eval.py [--url http://localhost:8000] [--n 48] [--conc 8] [--lengths 8000,16000,24000,30000] --out FILE
"""
import argparse, asyncio, json, random, re, time
import httpx

FILLER = ("The facilities team reviewed the maintenance schedule and noted that corridor lighting, ventilation "
          "filters and emergency signage should be checked every quarter. ").split()

def make_prompt(rnd, n_tokens, n_targets=4, update_frac=0.25):
    """Registry of `Room <code>: access PIN is <pin>.` records + filler; a fraction of rooms get a later
    `Update: Room <code> PIN changed to <pin>.` line. Targets are drawn mostly from updated rooms; the correct
    answer is each target's LATEST PIN (long-range tracking, not just lookup)."""
    records, seen = [], set()
    n_records = max(40, int(n_tokens / 14 * 0.5))
    while len(records) < n_records:
        code = f"{rnd.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rnd.randint(100, 999)}"
        if code not in seen:
            seen.add(code); records.append([code, f"{rnd.randint(1000, 9999)}"])
    n_upd = int(len(records) * update_frac)
    first_half = records[: len(records) // 2]
    updated = rnd.sample(first_half, min(n_upd, len(first_half)))
    events = [("set", c, p) for c, p in records]
    latest = {c: p for c, p in records}
    for c, _ in updated:  # insert the update somewhere after the original record
        pos = rnd.randint(len(records) // 2, len(events))
        newpin = f"{rnd.randint(1000, 9999)}"
        events.insert(pos, ("upd", c, newpin)); latest[c] = newpin
    filler_words = int(n_tokens * 0.45 / 1.3)
    per_gap = max(1, filler_words // len(events))
    lines = []
    for kind, c, p in events:
        lines.append(f"Room {c}: access PIN is {p}." if kind == "set" else f"Update: Room {c} PIN changed to {p}.")
        lines.append(" ".join(rnd.choice(FILLER) for _ in range(per_gap)))
    upd_codes = [c for c, _ in updated]
    k_upd = min(len(upd_codes), n_targets - 1)
    tcodes = rnd.sample(upd_codes, k_upd) + rnd.sample([c for c, _ in records if c not in upd_codes], n_targets - k_upd)
    rnd.shuffle(tcodes)
    targets = [(c, latest[c]) for c in tcodes]
    fmt = ", ".join(f"{c}=<PIN>" for c in tcodes)
    q = (f"Using only the registry above (later updates override earlier PINs), what are the current access PINs for rooms "
         f"{', '.join(tcodes)}? Answer in the form: {fmt}.")
    return "Building access registry:\n\n" + "\n".join(lines) + "\n\n" + q, targets

async def one(client, url, model, prompt, sem):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0, "max_tokens": 8192}
    content, usage, t0 = [], None, time.time()
    async with sem:
        async with client.stream("POST", url + "/v1/chat/completions", json=body, timeout=900) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("usage"):
                    usage = d["usage"]
                for c in d.get("choices", []):
                    content.append(c.get("delta", {}).get("content") or "")
    return "".join(content), usage, time.time() - t0

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000"); ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--conc", type=int, default=8); ap.add_argument("--lengths", default="8000,14000,20000,26000")
    ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    lengths = [int(x) for x in a.lengths.split(",")]
    cases = [(lengths[i % len(lengths)],) + make_prompt(rnd, lengths[i % len(lengths)]) for i in range(a.n)]
    async with httpx.AsyncClient() as client:
        model = (await client.get(a.url + "/v1/models")).json()["data"][0]["id"]
        sem = asyncio.Semaphore(a.conc)
        res = await asyncio.gather(*[one(client, a.url, model, p, sem) for _, p, _ in cases])
    rows, by_len = [], {}
    for (L, _, targets), (text, usage, dt) in zip(cases, res):
        ok_each = [bool(re.search(rf"{code}\s*[=:]\s*{pin}\b", text)) for code, pin in targets]
        ok = all(ok_each)
        by_len.setdefault(L, []).append(ok)
        rows.append({"len": L, "prompt_tokens": (usage or {}).get("prompt_tokens"), "completion_tokens": (usage or {}).get("completion_tokens"),
                     "targets": targets, "correct": ok, "partial": sum(ok_each), "answer_tail": text[-200:], "secs": round(dt, 1)})
    summ = {"model": model, "n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "pin_recall": sum(r["partial"] for r in rows) / sum(len(r["targets"]) for r in rows),
            "by_len": {str(L): sum(v) / len(v) for L, v in sorted(by_len.items())},
            "mean_prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows) / len(rows)}
    json.dump({"summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print(json.dumps(summ))

asyncio.run(main())
