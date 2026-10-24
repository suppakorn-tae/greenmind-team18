"""Pure-function tests for gm.history (Agent H). No store, no official import."""
import json
from pathlib import Path

import pytest

from gm import history

ROOT = Path(__file__).resolve().parent.parent
BASE_SUMMARY = json.loads((ROOT / "fixtures" / "base" / "summary.json").read_text())

SERVER = {
    "model": "M",
    "vllm_cmdline": ("/x/bin/python3 /x/bin/vllm serve M --port 8000 "
                     "--max-model-len 32768 --dtype bfloat16"),
    "nvidia_smi": {"power": {"limit": 300.0}},
}


def make_run(rid="r1", track="val", label="run", status="done",
             started="2026-10-24T12:00:00.123456", is_baseline=False,
             server=None, summary=None, score=None, notes=""):
    return {"id": rid, "track": track, "label": label, "notes": notes,
            "status": status, "started_at": started, "ended_at": None,
            "is_baseline": is_baseline, "server": server, "settings": {},
            "summary": summary, "score": score}


def make_official(oid=1, track="val", started="2026-10-24T11:00:00", model="M",
                  label="off", notes="", S=None, A=None, R=None, s_E=None,
                  s_T=None, s_L=None, eta_vs_base=None, j_per_req=None,
                  p95_ttft_s=None, p95_tpot_ms=None, slo_met=None, slo_total=None,
                  accuracy_pct=None, local_run_id=None):
    return {"id": oid, "track": track, "started_at": started, "model": model,
            "label": label, "notes": notes, "S": S, "A": A, "R": R, "s_E": s_E,
            "s_T": s_T, "s_L": s_L, "eta_vs_base": eta_vs_base,
            "j_per_req": j_per_req, "p95_ttft_s": p95_ttft_s,
            "p95_tpot_ms": p95_tpot_ms, "slo_met": slo_met, "slo_total": slo_total,
            "accuracy_pct": accuracy_pct, "local_run_id": local_run_id,
            "created_at": started}


# ── local_row ─────────────────────────────────────────────────────────────────
def test_local_row_mapping():
    run = make_run(rid="r1", label="cand", server=SERVER, summary=BASE_SUMMARY,
                   score={"S": 42.5, "A": 0.9, "R": 1.2, "s_E": 0.8, "eta_ratio": 1.1})
    row = history.local_row(run)
    assert row["key"] == "local:r1"
    assert row["source"] == "local"
    assert row["id"] == "r1"
    assert row["track"] == "val"
    assert row["label"] == "cand"
    assert row["started"] == "2026-10-24T12:00:00"  # microseconds normalised
    assert row["status"] == "done"
    assert row["is_baseline"] is False
    assert row["pair_key"] is None
    # score mapping
    assert row["S"] == 42.5 and row["A"] == 0.9 and row["R"] == 1.2
    assert row["s_E"] == 0.8 and row["eta_vs_base"] == 1.1
    # summary mapping
    assert row["s_T"] == BASE_SUMMARY["s_T"]
    assert row["s_L"] == BASE_SUMMARY["s_L"]
    assert row["j_per_req"] == BASE_SUMMARY["j_per_req"]
    assert row["p95_ttft_s"] == BASE_SUMMARY["p95_ttft_s"]
    assert row["p95_tpot_ms"] == BASE_SUMMARY["p95_tpot_ms"]
    assert row["slo_met"] == 87 and row["slo_total"] == 100
    assert row["accuracy_pct"] == 89.0
    assert row["slo_turn_met"] == 129 and row["slo_turn_total"] == 142
    assert row["eta"] == BASE_SUMMARY["eta"]
    # server fields
    assert row["model"] == "M"
    assert row["power_limit"] == 300.0
    assert row["flags"] == "serve M --port 8000 --max-model-len 32768 --dtype bfloat16"
    assert row["server_summary"] == "M | 300.0 W | --max-model-len 32768 --dtype bfloat16"
    assert row["group"] is None and row["repeat_k"] is None


def test_local_row_fixture_baseline_score():
    run = make_run(rid="r0", label="base", is_baseline=True, server=SERVER,
                   summary=BASE_SUMMARY,
                   score={"S": 0.0, "is_baseline": True, "eta_ratio": 1.0,
                          "R": 1.0, "A": 1.0, "s_E": 0.0})
    row = history.local_row(run)
    assert row["is_baseline"] is True
    assert row["S"] == 0.0 and row["eta_vs_base"] == 1.0
    assert row["R"] == 1.0 and row["A"] == 1.0 and row["s_E"] == 0.0
    assert row["j_per_req"] == pytest.approx(349.58013040093147)


def test_local_row_non_done_metrics_none():
    run = make_run(rid="r2", label="wip", status="running", server=SERVER,
                   summary={"eta": 0.5}, score={"S": 9.0, "eta_ratio": 1.0})
    row = history.local_row(run)
    assert row["status"] == "running"
    for k in ("S", "A", "R", "s_E", "eta_vs_base", "eta", "j_per_req",
              "slo_met", "slo_total", "slo_turn_met", "slo_turn_total"):
        assert row[k] is None, k
    # label-derived fields still set
    run3 = make_run(rid="r3", label="X#2", status="running")
    row3 = history.local_row(run3)
    assert row3["group"] == "X" and row3["repeat_k"] == 2
    # no server at all
    assert row3["flags"] is None
    assert row3["power_limit"] is None
    assert row3["server_summary"] is None


def test_flags_and_server_summary_variants():
    # contract example: only --dtype is a key flag
    srv = {"model": "M",
           "vllm_cmdline": "/x/bin/python3 /x/bin/vllm serve M --port 8000 --dtype bfloat16"}
    row = history.local_row(make_run(rid="f1", server=srv))
    assert row["flags"] == "serve M --port 8000 --dtype bfloat16"
    assert row["server_summary"] == "M | --dtype bfloat16"
    # many key flags incl. boolean flag, kept in cmdline order
    srv2 = {"model": "CM",
            "vllm_cmdline": ("vllm serve CM --enable-prefix-caching "
                             "--max-num-seqs 64 --gpu-memory-utilization 0.9"),
            "nvidia_smi": {"power": {"limit": 250}}}
    row2 = history.local_row(make_run(rid="f2", server=srv2))
    assert row2["flags"] == ("serve CM --enable-prefix-caching --max-num-seqs 64 "
                             "--gpu-memory-utilization 0.9")
    assert row2["server_summary"] == ("CM | 250 W | --enable-prefix-caching "
                                      "--max-num-seqs 64 --gpu-memory-utilization 0.9")
    # no vllm token / no cmdline
    assert history.local_row(make_run(rid="f3", server={"vllm_cmdline": "python3 x.py"}))["flags"] is None
    assert history.local_row(make_run(rid="f4", server=None))["flags"] is None


# ── official_row ──────────────────────────────────────────────────────────────
def test_official_row_mapping():
    o = make_official(oid=7, label="off base", eta_vs_base=1.0, j_per_req=467.0,
                      p95_ttft_s=0.6, p95_tpot_ms=41.0, slo_met=64, slo_total=100,
                      accuracy_pct=88.0, S=10.0, local_run_id="r1")
    row = history.official_row(o, local_baseline_id="r1", pair_key="local:r1")
    assert row["key"] == "official:7"
    assert row["source"] == "official"
    assert row["id"] == 7
    assert row["status"] == "done"
    assert row["is_baseline"] is True  # linked to the local baseline id
    assert row["pair_key"] == "local:r1"
    assert row["eta_vs_base"] == 1.0 and row["j_per_req"] == 467.0
    assert row["slo_met"] == 64 and row["slo_total"] == 100
    assert row["s_T"] is None and row["eta"] is None
    assert row["slo_turn_met"] is None
    assert row["model"] == "M" and row["flags"] is None and row["power_limit"] is None
    assert row["server_summary"] == "M"
    # fallback baseline id also marks baseline
    row2 = history.official_row(make_official(oid=8, label="b2"),
                                baseline_official_id=8)
    assert row2["is_baseline"] is True
    assert history.official_row(make_official(oid=9))["is_baseline"] is False


# ── build_rows wiring ─────────────────────────────────────────────────────────
def test_build_rows_newest_first():
    runs = [make_run("old", started="2026-10-20T10:00:00.999999"),
            make_run("new", started="2026-10-24T10:00:00.000001")]
    officials = [make_official(1, started="2026-10-22T00:00:00")]
    rows = history.build_rows(runs, officials)
    assert [r["key"] for r in rows] == ["local:new", "official:1", "local:old"]


def test_build_rows_baselines_and_pair_keys():
    runs = [
        make_run("r-base", label="base", is_baseline=True,
                 started="2026-10-24T10:00:00"),
        make_run("r-cand", label="cand", started="2026-10-24T12:00:00"),
    ]
    officials = [
        make_official(1, started="2026-10-24T09:00:00", label="baseline official"),
        make_official(2, started="2026-10-24T11:00:00", label="linked base",
                      local_run_id="r-base"),
        make_official(3, started="2026-10-24T12:30:00", label="linked cand",
                      local_run_id="r-cand"),
    ]
    rows = history.build_rows(runs, officials)
    by = {r["key"]: r for r in rows}
    assert by["local:r-base"]["is_baseline"] is True
    assert by["official:2"]["is_baseline"] is True   # linked to local baseline
    assert by["official:1"]["is_baseline"] is False  # fallback not used
    assert by["official:3"]["is_baseline"] is False
    assert by["official:2"]["pair_key"] == "local:r-base"
    assert by["local:r-base"]["pair_key"] == "official:2"
    assert by["local:r-cand"]["pair_key"] == "official:3"
    assert by["official:1"]["pair_key"] is None


def test_build_rows_official_baseline_fallback():
    runs = [make_run("r1", label="x", started="2026-10-24T12:00:00")]
    officials = [
        make_official(1, started="2026-10-24T09:00:00", label="baseline A"),
        make_official(2, started="2026-10-24T10:00:00", label="baseline B"),
        make_official(3, started="2026-10-24T08:00:00", label="other"),
    ]
    rows = history.build_rows(runs, officials)
    by = {r["key"]: r for r in rows}
    assert by["official:1"]["is_baseline"] is True    # earliest 'baseline*'
    assert by["official:2"]["is_baseline"] is False
    assert by["official:3"]["is_baseline"] is False


def test_local_pair_key_is_newest_official():
    runs = [make_run("r1", started="2026-10-24T12:00:00")]
    officials = [make_official(1, started="2026-10-24T10:00:00", local_run_id="r1"),
                 make_official(2, started="2026-10-24T11:00:00", local_run_id="r1")]
    rows = history.build_rows(runs, officials)
    local = next(r for r in rows if r["source"] == "local")
    assert local["pair_key"] == "official:2"
    # dangling local_run_id: official has no pair, local untouched
    rows2 = history.build_rows([make_run("r2")],
                               [make_official(1, local_run_id="ghost")])
    assert next(r for r in rows2 if r["source"] == "local")["pair_key"] is None
    assert next(r for r in rows2 if r["source"] == "official")["pair_key"] == "local:ghost"


def test_build_rows_track_filter():
    runs = [make_run("a", track="val"), make_run("b", track="final")]
    officials = [make_official(1, track="val"), make_official(2, track="final")]
    rows = history.build_rows(runs, officials, track="val")
    assert {r["key"] for r in rows} == {"local:a", "official:1"}


# ── repeats / groups / noise ──────────────────────────────────────────────────
def test_parse_repeat():
    assert history.parse_repeat("X#3") == ("X", 3)
    assert history.parse_repeat("modelA#10") == ("modelA", 10)
    assert history.parse_repeat("a#1#2") == ("a#1", 2)
    assert history.parse_repeat("plain") == (None, None)
    assert history.parse_repeat("X#") == (None, None)
    assert history.parse_repeat(None) == (None, None)


def _rep_run(rid, label, eta, evb, slo_met, slo_total, status="done",
             started="2026-10-24T12:00:00", track="val"):
    summary = None
    score = None
    if status == "done":
        summary = {"eta": eta, "j_per_req": 300.0, "p95_ttft_s": 0.5,
                   "p95_tpot_ms": 40.0, "slo_met": slo_met, "slo_total": slo_total,
                   "accuracy_pct": 90.0}
        score = {"S": 10.0, "eta_ratio": evb, "A": 1.0, "R": 1.0, "s_E": 1.0}
    return make_run(rid, label=label, status=status, started=started,
                    summary=summary, score=score, track=track)


def test_group_repeats():
    runs = [
        _rep_run("a", "X#1", 0.001, 1.0, 90, 100),
        _rep_run("b", "X#2", 0.002, 1.1, 80, 100),
        _rep_run("c", "X#3", 0.003, 1.2, 70, 100),
        _rep_run("d", "Y#1", 0.001, 1.0, 90, 100),       # lone member: no group
        _rep_run("e", "X#4", 0.0, 0.0, 0, 0, status="running"),  # not done
    ]
    rows = history.build_rows(runs, [])
    groups = history.group_repeats(rows)
    assert len(groups) == 1
    g = groups[0]
    assert g["track"] == "val" and g["group"] == "X" and g["n"] == 3
    assert set(g["members"]) == {"local:a", "local:b", "local:c"}
    assert g["mean"]["eta"] == pytest.approx(0.002)
    assert g["min"]["eta"] == 0.001 and g["max"]["eta"] == 0.003
    assert g["spread_eta"] == pytest.approx(1.0)
    assert g["mean"]["slo_met_frac"] == pytest.approx(0.8)
    assert g["mean"]["eta_vs_base"] == pytest.approx(1.1)
    assert g["mean"]["S"] == 10.0
    assert g["mean"]["j_per_req"] == 300.0
    # different tracks do not merge
    g2 = history.group_repeats([
        history.local_row(_rep_run("p", "X#1", 1.0, 1.0, 1, 1, track="val")),
        history.local_row(_rep_run("q", "X#2", 2.0, 1.0, 1, 1, track="final")),
    ])
    assert g2 == []


def test_noise_spread_default_vs_observed():
    assert history.noise_spread([]) == {"spread": 0.10, "observed": False, "groups": 0}
    assert history.noise_spread([], default=0.05)["spread"] == 0.05
    runs = [_rep_run("a", "X#1", 0.001, 1.0, 90, 100),
            _rep_run("b", "X#2", 0.002, 1.1, 80, 100)]
    rows = history.build_rows(runs, [])
    ns = history.noise_spread(rows)
    assert ns == {"spread": pytest.approx(1.0 / 1.5), "observed": True, "groups": 1}


def test_within_noise():
    assert history.within_noise(1.0, 1.0, 0.1) is True
    assert history.within_noise(11, 10, 0.1) is True      # exact boundary
    assert history.within_noise(12, 10, 0.1) is False
    assert history.within_noise(None, 1.0, 0.1) is False
    assert history.within_noise(1.0, None, 0.1) is False
    assert history.within_noise(1.0, 0.0, 0.1) is False   # tiny denominator
    assert history.within_noise(1.0, 1.0, None) is False


def test_mark_noise():
    runs = [
        make_run("r-base", label="base", is_baseline=True,
                 started="2026-10-24T10:00:00",
                 summary={"eta": 0.001},
                 score={"S": 0.0, "eta_ratio": 1.0}),
        make_run("r-near", label="near", started="2026-10-24T11:00:00",
                 summary={"eta": 0.0011}, score={"S": 5.0, "eta_ratio": 1.05}),
        make_run("r-far", label="far", started="2026-10-24T12:00:00",
                 summary={"eta": 0.0015}, score={"S": 6.0, "eta_ratio": 1.5}),
        make_run("r-run", label="wip", started="2026-10-24T13:00:00",
                 status="running"),
    ]
    rows = history.build_rows(runs, [])
    history.mark_noise(rows, 0.10)
    by = {r["key"]: r for r in rows}
    assert by["local:r-base"]["within_noise"] is None     # baseline row
    assert by["local:r-near"]["within_noise"] is True     # 5% off 1.0
    assert by["local:r-far"]["within_noise"] is False     # 50% off
    assert by["local:r-run"]["within_noise"] is False     # no metrics
    # no baseline in track -> reference 1.0
    rows2 = history.build_rows(
        [make_run("x", started="2026-10-24T10:00:00",
                  summary={"eta": 1.0}, score={"S": 1.0, "eta_ratio": 1.04}),
         make_run("y", started="2026-10-24T11:00:00",
                  summary={"eta": 1.0}, score={"S": 2.0, "eta_ratio": 1.4})], [])
    history.mark_noise(rows2, 0.10)
    by2 = {r["key"]: r for r in rows2}
    assert by2["local:x"]["within_noise"] is True
    assert by2["local:y"]["within_noise"] is False


# ── calibration ───────────────────────────────────────────────────────────────
def _calib_runs():
    l1 = make_run("l1", label="cand A", started="2026-10-24T12:00:00",
                  summary={"j_per_req": 500.0, "slo_met": 90, "slo_total": 100,
                           "p95_ttft_s": 0.5, "p95_tpot_ms": 40.0,
                           "accuracy_pct": 90.0},
                  score={"S": 10.0, "eta_ratio": 1.2})
    l2 = make_run("l2", label="cand B", started="2026-10-24T12:30:00",
                  summary={"j_per_req": 500.0, "slo_met": 80, "slo_total": 100,
                           "p95_ttft_s": 0.4, "p95_tpot_ms": 30.0,
                           "accuracy_pct": 95.0},
                  score={"S": 9.0, "eta_ratio": 0.9})
    l3 = make_run("l3", label="broken", started="2026-10-24T12:40:00",
                  status="failed",
                  summary={"j_per_req": 100.0},
                  score={"S": 1.0, "eta_ratio": 5.0})
    o1 = make_official(1, started="2026-10-24T11:00:00", label="off A",
                       j_per_req=467.0, slo_met=64, slo_total=100,
                       p95_ttft_s=0.5, p95_tpot_ms=40.0, accuracy_pct=90.0,
                       eta_vs_base=1.1, local_run_id="l1")
    o2 = make_official(2, started="2026-10-24T11:30:00", label="off B",
                       j_per_req=400.0, slo_met=50, slo_total=100,
                       p95_ttft_s=0.4, p95_tpot_ms=30.0, accuracy_pct=None,
                       eta_vs_base=1.3, local_run_id="l2")
    o3 = make_official(3, started="2026-10-24T11:40:00", label="off broken",
                       j_per_req=100.0, eta_vs_base=5.0, local_run_id="l3")
    o4 = make_official(4, started="2026-10-24T11:50:00", label="off ghost",
                       eta_vs_base=1.0, local_run_id="no-such-run")
    return [l1, l2, l3], [o1, o2, o3, o4]


def test_calibration_flags_and_ratios():
    runs, officials = _calib_runs()
    cal = history.calibration(history.build_rows(runs, officials), "val")
    assert len(cal["pairs"]) == 2  # l3 not done, o4 dangling
    p1 = next(p for p in cal["pairs"] if p["local_key"] == "local:l1")
    m = {x["name"]: x for x in p1["metrics"]}
    assert [x["name"] for x in p1["metrics"]] == [
        "j_per_req", "slo_met_frac", "p95_ttft_s", "p95_tpot_ms",
        "accuracy_pct", "eta_vs_base"]
    assert m["j_per_req"]["ratio"] == pytest.approx(500 / 467)
    assert m["j_per_req"]["flag"] == "green"      # 1.071
    assert m["slo_met_frac"]["local"] == pytest.approx(0.9)
    assert m["slo_met_frac"]["official"] == pytest.approx(0.64)
    assert m["slo_met_frac"]["ratio"] == pytest.approx(0.9 / 0.64)
    assert m["slo_met_frac"]["flag"] == "red"     # 1.406
    assert m["p95_ttft_s"]["flag"] == "green"
    p2 = next(p for p in cal["pairs"] if p["local_key"] == "local:l2")
    m2 = {x["name"]: x for x in p2["metrics"]}
    assert m2["j_per_req"]["ratio"] == pytest.approx(1.25)
    assert m2["j_per_req"]["flag"] == "amber"     # 0.25 off
    assert m2["accuracy_pct"]["official"] is None
    assert m2["accuracy_pct"]["flag"] == "na"


def test_rank_agreement_disagree_and_agree():
    runs, officials = _calib_runs()
    cal = history.calibration(history.build_rows(runs, officials), "val")
    ra = cal["rank_agreement"]
    assert ra["n_pairs"] == 2
    # local: l1(1.2) > l2(0.9); official: o1(1.1) < o2(1.3) -> disagree
    assert ra == {"n_pairs": 2, "agree": 0, "total": 1,
                  "text": "0 of 1 pair-orderings agree"}
    # three agreeing pairs
    runs2 = [make_run(f"x{i}", started=f"2026-10-24T12:0{i}:00",
                      summary={"eta": 1.0},
                      score={"S": float(i), "eta_ratio": v})
             for i, v in enumerate([1.2, 1.0, 0.8])]
    off2 = [make_official(i + 1, started=f"2026-10-24T11:0{i}:00",
                          eta_vs_base=v, local_run_id=f"x{i}")
            for i, v in enumerate([1.1, 1.0, 0.9])]
    cal2 = history.calibration(history.build_rows(runs2, off2), "val")
    assert cal2["rank_agreement"] == {"n_pairs": 3, "agree": 3, "total": 3,
                                      "text": "3 of 3 pair-orderings agree"}


def test_rank_agreement_need_two():
    l = make_run("l1", started="2026-10-24T12:00:00",
                 summary={"j_per_req": 100.0},
                 score={"S": 1.0, "eta_ratio": 1.0})
    o = make_official(1, eta_vs_base=1.0, local_run_id="l1")
    cal = history.calibration(history.build_rows([l], [o]), "val")
    assert len(cal["pairs"]) == 1
    assert cal["rank_agreement"] == {"n_pairs": 1, "agree": 0, "total": 0,
                                     "text": "need >= 2 linked pairs"}
    assert history.calibration([], "val")["rank_agreement"]["text"] == "need >= 2 linked pairs"


# ── best_rows ─────────────────────────────────────────────────────────────────
def test_best_rows():
    runs = [
        make_run("b", label="base", is_baseline=True, started="2026-10-24T10:00:00",
                 score={"S": 0.0, "eta_ratio": 1.0}),
        make_run("c1", started="2026-10-24T11:00:00",
                 score={"S": 12.0, "eta_ratio": 1.0}),
        make_run("c2", started="2026-10-24T11:30:00",
                 score={"S": 12.0, "eta_ratio": 1.3}),   # S tie, better eta_vs_base
        make_run("c3", started="2026-10-24T11:45:00", status="running",
                 score={"S": 99.0}),                     # not done -> S None
    ]
    officials = [make_official(1, started="2026-10-24T09:00:00", S=5.0,
                               eta_vs_base=1.0),
                 make_official(2, started="2026-10-24T09:30:00", S=7.5,
                               eta_vs_base=0.9),
                 make_official(3, started="2026-10-24T09:45:00", S=None,
                               eta_vs_base=2.0)]         # S None -> ignored
    rows = history.build_rows(runs, officials)
    best = history.best_rows(rows, "val")
    assert best["local"]["key"] == "local:c2"
    assert best["official"]["key"] == "official:2"
    assert history.best_rows(rows, "final") == {"local": None, "official": None}


# ── format_history ────────────────────────────────────────────────────────────
def test_format_history_smoke():
    runs = [
        make_run("r-base", label="base", is_baseline=True,
                 started="2026-10-24T10:00:00", server=SERVER,
                 summary={"eta": 0.001, "j_per_req": 350, "p95_ttft_s": 0.54,
                          "p95_tpot_ms": 43.0, "slo_met": 87, "slo_total": 100,
                          "accuracy_pct": 89.0, "slo_turn": {"on_time": 129, "total": 142}},
                 score={"S": 0.0, "eta_ratio": 1.0, "A": 1.0, "R": 1.0, "s_E": 0.0}),
        make_run("r-cand", label="cand", started="2026-10-24T12:00:00",
                 server=SERVER,
                 summary={"eta": 0.0011, "j_per_req": 340, "p95_ttft_s": 0.6,
                          "p95_tpot_ms": 40.0, "slo_met": 90, "slo_total": 100,
                          "accuracy_pct": 91.0},
                 score={"S": 12.3, "eta_ratio": 1.1, "A": 0.95, "R": 1.2, "s_E": 0.8}),
    ]
    officials = [make_official(1, started="2026-10-24T09:00:00", label="off base",
                               S=4.0, j_per_req=467, p95_ttft_s=0.7,
                               p95_tpot_ms=41.0, slo_met=64, slo_total=100,
                               accuracy_pct=88.0, eta_vs_base=1.0,
                               local_run_id="r-base")]
    rows = history.build_rows(runs, officials)
    history.mark_noise(rows, 0.10)
    out = history.format_history(rows,
                                 groups=history.group_repeats(rows),
                                 spread=history.noise_spread(rows))
    lines = out.splitlines()
    # header + 3 rows + group/noise footer
    assert lines[0].startswith("SRC")
    assert "LABEL" in lines[0] and "CONFIG" in lines[0]
    body = "\n".join(lines[1:4])
    assert "cand" in body and "base" in body and "off base" in body
    assert "87/100" in body          # SLO met/total
    assert "12.3" in body            # S, 1 decimal
    assert "1.100" in body           # ratios, 3 decimals
    assert "350" in body              # J/REQ, 0 decimals
    assert "0.54" in body             # TTFT, 2 decimals
    assert "89.0" in body             # ACC%, 1 decimal
    assert "*" in body                # baseline marker
    # newest first: cand (12:00) before base (10:00) before official (09:00)
    # (LABEL column starts at index 9)
    assert lines[1][9:25].startswith("cand")
    assert lines[2][9:25].startswith("base")
    assert lines[3][9:25].startswith("off base")
    # footer: default noise spread (no repeat groups)
    assert "noise spread ±10.0% (default); rows marked ~ are within noise of the baseline" in out
    # config truncated to 60 chars max (CONFIG is the last column)
    cfg_col = lines[0].index("CONFIG")
    assert all(len(ln[cfg_col:]) <= 60 for ln in lines[1:4])
    # pair column shows counterpart label: local base row -> official label
    assert "off base" in lines[2]
    # no repeat groups -> no group section
    assert " x2:" not in out and " x3:" not in out


def test_format_history_all_none_metrics():
    rows = history.build_rows(
        [make_run("z", label="zombie", status="running", server=None,
                  summary=None, score=None),
         make_run("n", label="near", started="2026-10-24T11:00:00",
                  summary={"eta": 1.0}, score={"S": 1.0, "eta_ratio": 1.05})], [])
    out = history.format_history(rows)
    zline = [ln for ln in out.splitlines() if "zombie" in ln][0]
    assert zline.count("-") >= 8
    # no baseline in track -> ref 1.0: 1.05 within 10%, zombie (no metrics) not
    history.mark_noise(rows, 0.10)
    out2 = history.format_history(rows)
    assert "near~" in out2
    assert "zombie~" not in out2


def test_format_history_groups_and_observed_footer():
    runs = [_rep_run("a", "X#1", 0.001, 1.0, 90, 100),
            _rep_run("b", "X#2", 0.002, 1.1, 80, 100),
            _rep_run("c", "X#3", 0.003, 1.2, 70, 100)]
    rows = history.build_rows(runs, [])
    ns = history.noise_spread(rows)
    out = history.format_history(rows, groups=history.group_repeats(rows), spread=ns)
    assert "val:X x3: ETA/BASE mean 1.100 (1.000–1.200), S mean 10.0, J/REQ 300" in out
    assert f"noise spread ±{ns['spread'] * 100:.1f}% (observed)" in out
