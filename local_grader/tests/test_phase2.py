"""Official-table columns in the summary, official CLI round-trip, history CLI, run --repeat."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grader
from gm import config, energy, score, trace
from gm.store import Store
from test_e2e import FakeNVML  # noqa: E402  (same fake NVML as the e2e test)


def rec(i, turn, ok=True, ttft=0.2, cls="agent", phase="peak"):
    return {"id": i, "cls": cls, "phase": phase, "turn": turn, "ttft_any": ttft, "ttft_content": ttft, "span_any": 1.0,
            "span_content": 1.0, "tpot_chunk_p95": 0.01, "prompt_tokens": 100, "completion_tokens": 101,
            "finish": "stop", "has_usage": True, "error": None if ok else "boom"}


def summarize(records, items, J=1000.0):
    raw = {"ok": True, "idle_W": 50.0, "phase_J": {}, "windows": {w: {"J": J, "s": 10.0} for w in config.SETTING_CHOICES["ENERGY_WINDOW"]}}
    return score.summarize(records, items, raw, config.resolve_settings(), {"ramp": (0, 1), "peak": (1, 2), "quiet": (2, 3)})


def test_per_arrival_vs_per_turn_slo_met_and_other_official_columns():
    # item 0: two turns, second fails → arrival NOT on time, turns 1/2 ; item 1: single on-time turn ; item 2: single failing turn
    records = [rec(0, 0), rec(0, 1, ok=False), rec(1, 0, cls="chat"), rec(2, 0, ok=False, cls="chat")]
    items = [{"id": 0, "cls": "agent", "correct": True, "final_turn": 1}, {"id": 1, "cls": "chat", "correct": True, "final_turn": 0},
             {"id": 2, "cls": "chat", "correct": False, "final_turn": 0}]
    s = summarize(records, items, J=900.0)
    assert s["slo_turn"] == {"on_time": 2, "total": 4}
    assert s["slo_arrival"] == {"on_time": 1, "total": 3}
    assert (s["slo_met"], s["slo_total"]) == (1, 3)                      # official-comparable = per arrival
    assert s["j_per_req"] == pytest.approx(900.0 / 3)                    # window joules ÷ arrivals
    assert s["p95_ttft_s"] == pytest.approx(0.2)                          # raw seconds, not a ratio
    # micro accuracy: item 2 hard-failed (error) → wrong; items 0 (agent final turn failed with error → wrong) and 1 right
    assert s["accuracy_pct"] == pytest.approx(100 * 1 / 3)


def test_columns_present_and_none_without_nvml():
    records, items = [rec(0, 0)], [{"id": 0, "cls": "chat", "correct": True, "final_turn": 0}]
    s = score.summarize(records, items, None, config.resolve_settings(), {"ramp": (0, 1), "peak": (1, 2), "quiet": (2, 3)})
    assert s["j_per_req"] is None and s["slo_met"] == 1 and s["accuracy_pct"] == 100.0


def test_official_cli_roundtrip_and_history(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GRADER_HOME", str(tmp_path))
    st = Store(tmp_path)
    rid = st.create_run("val", "base", "", {"model": "m"}, config.resolve_settings())
    st.finish_run(rid, "done", {"eta": 0.003, "slo_met": 90, "slo_total": 100, "j_per_req": 300.0, "accuracy_pct": 90.0,
                                "p95_ttft_s": 1.0, "p95_tpot_ms": 30.0, "s_T": 0.9, "s_L": 0.9})
    st.set_baseline(rid)
    st.set_score(rid, {"S": 0.0, "is_baseline": True, "eta_ratio": 1.0, "R": 1.0, "A": 1.0, "s_E": 0.0})
    args = ["official", "add", "--track", "val", "--started", "2026-10-24 04:50:00", "--model", "M", "--S", "0.0", "--A", "1.00",
            "--R", "1.000", "--sE", "0.00", "--sT", "0.51", "--sL", "0.59", "--eta-vs-base", "0.89", "--j-per-req", "467",
            "--p95-ttft", "2.06", "--p95-tpot-ms", "48", "--slo-met", "64", "--slo-total", "100", "--accuracy", "87.2",
            "--label", "baseline_bf16_default"]
    assert grader.main(args) == 0
    assert grader.main(["official", "link", "1", "base"]) == 0
    assert grader.main(["official", "list", "--track", "val"]) == 0
    out = capsys.readouterr().out
    assert "baseline_bf16_default" in out and "467" in out and rid in out
    assert grader.main(["history", "--track", "val"]) == 0
    out = capsys.readouterr().out
    assert "official" in out and "local" in out and "baseline_bf16_default" in out
    with pytest.raises(SystemExit):
        grader.main(["official", "add", "--track", "bogus"])


def test_run_repeat_labels_and_noise_grouping(mock_server, tmp_path, monkeypatch):
    """`run --repeat 3` → three runs labelled L#1..L#3; history groups them (short 4 s slice of the val trace)."""
    fake = FakeNVML(200.0)
    monkeypatch.setattr(energy, "pynvml", fake)
    monkeypatch.setattr(energy, "server_context", lambda: {})
    monkeypatch.setenv("GRADER_HOME", str(tmp_path))
    real_load = trace.load

    def short_load(track, root=None):
        items, bounds, meta = real_load(track, root)
        return [it for it in items if it["t"] < 4], bounds, meta
    monkeypatch.setattr(trace, "load", short_load)
    rc = grader.main(["run", "--track", "val", "--label", "cfgA", "--repeat", "3", "--url", mock_server.url, "--settle", "0.5", "--warmup", "1"])
    assert rc == 0
    labels = sorted(r["label"] for r in Store(tmp_path).list_runs("val"))
    assert labels == ["cfgA#1", "cfgA#2", "cfgA#3"]
    from gm import history
    rows = history.load_rows(Store(tmp_path), "val")
    groups = history.group_repeats(rows)
    assert len(groups) == 1 and groups[0]["n"] == 3 and groups[0]["group"] == "cfgA"
    assert history.noise_spread(rows)["observed"] is True
    prog = Store(tmp_path).run_dir(Store(tmp_path).list_runs("val")[0]["id"]) / "progress.json"
    assert prog.exists()
