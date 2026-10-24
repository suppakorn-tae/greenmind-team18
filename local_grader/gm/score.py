"""s_E, s_T, s_L, A, S (spec §10 — math ported unchanged from greenbench) + the run summary builder."""
import math

from . import config, slo, accuracy
from .config import CHECKED, K_ENERGY, TPOT_SLO


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def _smooth(x):
    return 3 * x * x - 2 * x ** 3


def p95(xs):
    """Nearest-rank p95 (robust to inf)."""
    xs = sorted(xs)
    if not xs:
        return None
    return xs[max(0, math.ceil(0.95 * len(xs)) - 1)]


def p50(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def accuracy_factor(acc, base_acc):
    """→ (A, R, weakest, class_factor). weakest = (class, ratio) over checked classes; None if no class qualifies."""
    classes = [c for c in base_acc if c in acc and base_acc[c] > 0]
    if not classes:
        return None, None, None, None
    R = (sum(acc[c] for c in classes) / len(classes)) / (sum(base_acc[c] for c in classes) / len(classes))
    cf, weakest = 1.0, None
    for c in classes:
        if c in CHECKED:
            ratio = acc[c] / base_acc[c]
            f = _smooth(_clip((ratio - 0.50) / 0.25))
            if weakest is None or ratio < weakest[1]:
                weakest = (c, round(ratio, 3))
            cf = min(cf, f)
    A = max(0.05, _smooth(_clip((R - 0.85) / 0.13)) * cf)
    return A, R, weakest, cf


def s_energy(ratio):
    return _clip(math.log(ratio) / math.log(K_ENERGY)) if ratio > 0 else 0.0


def score(summary, baseline):
    """Round-1 score vs a baseline summary. {} when either side lacks energy (η) data."""
    if not baseline or not baseline.get("eta") or not summary.get("eta"):
        return {}
    ratio = summary["eta"] / baseline["eta"]
    sE = s_energy(ratio)
    A, R, weakest, cf = accuracy_factor(summary["class_acc"], baseline["class_acc"])
    A = 1.0 if A is None else A
    sT, sL = summary["s_T"], summary["s_L"]
    S = 100 * A * (sE ** 0.6) * (sT ** 0.2) * (sL ** 0.2)
    class_ratio = {c: summary["class_acc"][c] / baseline["class_acc"][c]
                   for c in baseline["class_acc"] if c in summary["class_acc"] and baseline["class_acc"][c] > 0}
    return {"eta_ratio": ratio, "s_E": sE, "R": R, "A": A, "class_factor": cf,
            "weakest_class": list(weakest) if weakest else None, "class_ratio": class_ratio, "S": S}


def _group(rows):
    ratios = [r["ttft_ratio"] for r in rows]
    tpots = [r["tpot"] for r in rows if r["tpot"] is not None]
    cts = [r["completion_tokens"] for r in rows if r["completion_tokens"]]
    fails = {}
    for r in rows:
        for f in r["fails"]:
            fails[f] = fails.get(f, 0) + 1
    return {"n": len(rows), "on_time": sum(r["slo_ok"] for r in rows), "fails": fails,
            "p50_ttft_ratio": p50(ratios), "p95_ttft_ratio": p95(ratios),
            "p50_tpot_ms": p50(tpots) * 1000 if tpots else None, "p95_tpot_ms": p95(tpots) * 1000 if tpots else None,
            "completion_tokens_mean": sum(cts) / len(cts) if cts else None, "completion_tokens_p95": p95(cts),
            "cutoffs": sum("cutoff" in r["fails"] for r in rows)}


def summarize(records, items, energy_raw, settings, bounds, launch_stats=None):
    """records: raw rows of records.jsonl; items: rows of items.jsonl; energy_raw: see grader.run (may be None
    without NVML); bounds: {phase: (a, b)} → summary dict (schema: INTERFACES.md)."""
    rows = slo.effective(records, settings)
    n = len(rows)
    ok = sum(r["slo_ok"] for r in rows)
    peak = [r for r in rows if r["phase"] == "peak"]
    ttft_ratios = [r["ttft_ratio"] for r in rows]
    tpots = [r["tpot"] for r in rows if r["tpot"] is not None]
    pr, pt = p95(ttft_ratios), p95(tpots)
    sl_ttft = 0.0 if not pr or math.isinf(pr) else min(1.0, 1.0 / pr)
    sl_tpot = 0.0 if not pt else min(TPOT_SLO / pt, 1.0)
    final_fails = {}
    by_key = {(r["id"], r.get("turn", 0)): slo.evaluate(r, settings) for r in records}
    for it in items:
        final_fails[it["id"]] = by_key.get((it["id"], it.get("final_turn", 0)), [])
    class_acc, class_n = accuracy.class_accuracy(items, final_fails, settings)
    g = _group(rows)
    # official-table columns: per-turn vs per-arrival SLO-met (an episode is on time only if all its turns are)
    turn_rows = slo.effective(records, {**settings, "EPISODE_TURNS_SCORED": "each_turn"})
    arr_rows = slo.effective(records, {**settings, "EPISODE_TURNS_SCORED": "episode_once"})
    slo_turn = {"on_time": sum(r["slo_ok"] for r in turn_rows), "total": len(turn_rows)}
    slo_arrival = {"on_time": sum(r["slo_ok"] for r in arr_rows), "total": len(arr_rows)}
    raw_ttfts = [r["ttft"] for r in turn_rows if r["ttft"] is not None]
    n_acc = sum(class_n.values())
    s = {
        "requests": n, "on_time": ok, "on_time_frac": ok / n if n else 0.0, "fail_reasons": g["fails"],
        "s_T": sum(r["slo_ok"] for r in peak) / len(peak) if peak else 0.0,
        "s_L": math.sqrt(sl_ttft * sl_tpot), "s_L_ttft": sl_ttft, "s_L_tpot": sl_tpot,
        "p50_ttft_over_slo": g["p50_ttft_ratio"], "p95_ttft_over_slo": pr,
        "p50_tpot_ms": g["p50_tpot_ms"], "p95_tpot_ms": pt * 1000 if pt else None,
        "completion_tokens_mean": g["completion_tokens_mean"], "completion_tokens_p95": g["completion_tokens_p95"],
        "total_completion_tokens": sum(r["completion_tokens"] or 0 for r in rows), "cutoffs": g["cutoffs"],
        "by_phase": {p: _group([r for r in rows if r["phase"] == p]) for p in config.PHASES},
        "by_class": {c: _group([r for r in rows if r["cls"] == c]) for c in config.CLASS_MIX},
        "class_acc": class_acc, "class_n": class_n,
        "macro_acc": sum(class_acc.values()) / len(class_acc) if class_acc else None,
        "bounds": {k: list(v) for k, v in bounds.items()}, "launch": launch_stats or {},
        "n_arrivals": len(items), "slo_turn": slo_turn, "slo_arrival": slo_arrival,
        "slo_met": slo_arrival["on_time"], "slo_total": slo_arrival["total"],   # official-comparable (per arrival)
        "p95_ttft_s": p95(raw_ttfts),
        "accuracy_pct": 100.0 * sum(class_acc[c] * class_n[c] for c in class_acc) / n_acc if n_acc else None,
        "j_per_req": None,
        "energy_raw": energy_raw, "energy_ok": bool(energy_raw and energy_raw.get("ok")),
        "energy_J": None, "window_s": None, "avg_power_W": None, "idle_W": None, "eta": None,
        "tokens_per_J": None, "phase_energy_J": None,
    }
    if s["energy_ok"]:
        w = energy_raw["windows"][settings["ENERGY_WINDOW"]]
        s["energy_J"], s["window_s"] = w["J"], w["s"]
        s["avg_power_W"] = w["J"] / w["s"] if w["s"] else None
        s["idle_W"] = energy_raw.get("idle_W")
        s["eta"] = ok / w["J"] if w["J"] else None
        s["tokens_per_J"] = s["total_completion_tokens"] / w["J"] if w["J"] else None
        s["j_per_req"] = w["J"] / len(items) if items else None   # window joules ÷ arrivals (official counts arrivals)
        s["phase_energy_J"] = energy_raw.get("phase_J")
    return s
