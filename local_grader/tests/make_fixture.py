"""Build a synthetic but schema-correct run folder (records/items/power/summary) from the val trace.
Used by report/store tests:  make_fixture(dir, seed, power_w, slow) → summary dict."""
import json, random, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gm import config, score, trace


def make_fixture(out_dir, seed=1, power_w=300.0, slow=0.0, settings=None):
    settings = config.resolve_settings(settings)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    items_t, bounds = trace.build_trace(100, 0)
    records, items = [], []
    for it in items_t:
        turns = rng.randint(2, 4) if it["episode"] else 1
        t = it["t"]
        for k in range(turns):
            pt = len(json.dumps(it["messages"])) // 4 + 80 * k
            ct = rng.randint(200, 2500)
            ttft = rng.uniform(0.1, 0.6) + slow * rng.random() * 2
            tpot = rng.uniform(0.012, 0.045) + slow * 0.03 * rng.random()
            records.append({"id": it["id"], "cls": it["cls"], "turn": k, "sched_t": it["t"], "sent": round(t, 4),
                            "launch_lag": round(rng.random() * 0.01, 4) if k == 0 else None,
                            "phase": trace.phase_of(t, bounds), "ttft_any": ttft, "ttft_content": ttft + rng.random(),
                            "span_any": tpot * (ct - 1), "span_content": tpot * (ct - 1) * 0.5, "tpot_chunk_p95": tpot * 1.3,
                            "prompt_tokens": pt, "completion_tokens": ct, "finish": "tool_calls" if k < turns - 1 else "stop",
                            "has_usage": True, "error": None, "e2e": ttft + tpot * ct, "reasoning_chars": ct * 3,
                            "done": round(t + ttft + tpot * ct, 4), "n_tool_calls": 1 if k < turns - 1 else 0,
                            "answer_tail": "…the answer is \\boxed{42}"})
            t += ttft + tpot * ct
        items.append({"id": it["id"], "cls": it["cls"], "correct": rng.random() < 0.9, "turns": turns, "final_turn": turns - 1})
    last_done = max(r["done"] for r in records)
    # power trace: ramp up to peak, drop in quiet; joules integrated
    samples, j, t = [], 0.0, 0.0
    while t <= last_done + 0.05:
        ph = trace.phase_of(t, bounds)
        w = power_w * {"ramp": 0.7, "peak": 1.0, "quiet": 0.55, "drain": 0.4}[ph] + rng.uniform(-5, 5)
        samples.append((t, w, j)); j += w * 0.1; t += 0.1
    with (out / "power.csv").open("w") as f:
        f.write("t_s,watts,joules_cum\n")
        for t, w, j in samples:
            f.write(f"{t:.3f},{w:.1f},{j:.2f}\n")
    phase_J = {}
    for p in config.PHASES:
        a, b = (bounds[p] if p != "drain" else (bounds["quiet"][1], last_done))
        ja = next((s[2] for s in samples if s[0] >= a), samples[-1][2]); jb = next((s[2] for s in samples if s[0] >= min(b, last_done)), samples[-1][2])
        phase_J[p] = max(0.0, jb - ja)
    raw = {"ok": True, "idle_W": power_w * 0.2, "phase_J": phase_J,
           "windows": {"first_arrival_to_last_done": {"J": samples[-1][2], "s": last_done},
                       "first_send_to_last_done": {"J": samples[-1][2], "s": last_done}}}
    summary = score.summarize(records, items, raw, settings, bounds, {"max_launch_lag_ms": 10.0, "n_late_gt_100ms": 0, "max_loop_lag_ms": 12.0})
    for name, rows in (("records", records), ("items", items)):
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    s = make_fixture(sys.argv[1], seed=int(sys.argv[2]) if len(sys.argv) > 2 else 1,
                     power_w=float(sys.argv[3]) if len(sys.argv) > 3 else 300.0, slow=float(sys.argv[4]) if len(sys.argv) > 4 else 0.0)
    print(json.dumps({k: s[k] for k in ("requests", "on_time", "eta", "s_T", "s_L", "energy_J")}, indent=1))
