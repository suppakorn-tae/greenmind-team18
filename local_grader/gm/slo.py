"""Per-request SLO evaluation. Pure functions over the raw timing fields stored in records.jsonl, so `rescore`
can re-evaluate under different calibration settings (TTFT_FROM, TPOT_MODE, EPISODE_TURNS_SCORED)."""
import math

from . import config


def ttft_allowance(prompt_tokens):
    return config.TTFT_BASE_S + config.TTFT_PER_1K_S * prompt_tokens / 1000.0


def ttft_of(rec, settings):
    return rec.get("ttft_content") if settings["TTFT_FROM"] == "content_only" else rec.get("ttft_any")


def tpot_of(rec, settings):
    """Seconds per output token, or None when undefined (≤1 completion token / no token chunks)."""
    ct = rec.get("completion_tokens")
    if not ct or ct <= 1:
        return None
    if settings["TPOT_MODE"] == "p95_over_tokens":
        return rec.get("tpot_chunk_p95")
    span = rec.get("span_any")   # always the full token span: TTFT_FROM changes ttft only, and ct counts reasoning tokens too
    return None if span is None else span / (ct - 1)


def evaluate(rec, settings):
    """→ list of failing reasons (empty = on time). Order: no_stream, error, no_usage, cutoff, ttft, tpot."""
    fails = []
    ttft = ttft_of(rec, settings)
    pt, ct = rec.get("prompt_tokens"), rec.get("completion_tokens")
    if ttft is None:
        fails.append("no_stream")
    if rec.get("error"):
        fails.append("error")
    if not rec.get("has_usage"):
        fails.append("no_usage")
    if rec.get("finish") == "length" or (ct is not None and ct >= config.MAX_TOKENS):
        fails.append("cutoff")
    if ttft is not None and pt is not None and ttft > ttft_allowance(pt):
        fails.append("ttft")
    tpot = tpot_of(rec, settings)
    if tpot is not None and tpot > config.TPOT_SLO:
        fails.append("tpot")
    return fails


def _row(rec, settings):
    ttft, tpot, pt = ttft_of(rec, settings), tpot_of(rec, settings), rec.get("prompt_tokens")
    allow = ttft_allowance(pt) if pt is not None else None
    ratio = ttft / allow if (ttft is not None and allow) else math.inf   # no ttft → +inf (spec §10)
    fails = evaluate(rec, settings)
    return {"id": rec["id"], "cls": rec["cls"], "phase": rec["phase"], "turn": rec.get("turn", 0),
            "ttft": ttft, "ttft_slo": allow, "ttft_ratio": ratio, "tpot": tpot,
            "completion_tokens": rec.get("completion_tokens"), "finish": rec.get("finish"),
            "fails": fails, "slo_ok": not fails}


def effective(records, settings):
    """Scored requests under EPISODE_TURNS_SCORED. each_turn: one row per record. episode_once: one row per item
    (phase of its first turn; on time only if every turn is; worst ttft ratio / tpot; summed tokens)."""
    rows = [_row(r, settings) for r in records]
    if settings["EPISODE_TURNS_SCORED"] == "each_turn":
        return rows
    by_item = {}
    for r in rows:
        by_item.setdefault(r["id"], []).append(r)
    out = []
    for iid, rs in by_item.items():
        rs.sort(key=lambda r: r["turn"])
        first = rs[0]
        tpots = [r["tpot"] for r in rs if r["tpot"] is not None]
        fails = sorted({f for r in rs for f in r["fails"]}, key=lambda f: ["no_stream", "error", "no_usage", "cutoff", "ttft", "tpot"].index(f))
        out.append({**first, "ttft_ratio": max(r["ttft_ratio"] for r in rs), "tpot": max(tpots) if tpots else None,
                    "completion_tokens": sum(r["completion_tokens"] or 0 for r in rs),
                    "fails": fails, "slo_ok": not fails})
    return out
