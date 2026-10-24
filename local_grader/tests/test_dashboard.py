"""Agent S: tests for gm/dashboard.py (build_state, build_run_detail, serve/make_server, export_static)."""
import json
import re
import shutil
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

from gm import dashboard, official  # noqa: E402
from gm.store import Store  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "fixtures"

BASE_SCORE = {"S": 0.0, "is_baseline": True, "eta_ratio": 1.0, "R": 1.0, "A": 1.0, "s_E": 0.0}


# ── helpers ───────────────────────────────────────────────────────────────────
def _mk_run(store, track, label, fixture="base", server=None, score=None, summary_patch=None):
    """Copy fixtures/<fixture> into a fresh run dir, finish it done, attach score."""
    server = server if server is not None else {"model": f"mock-{fixture}"}
    rid = store.create_run(track, label, "", server, {"TTFT_FROM": "any"})
    dst = store.run_dir(rid)
    for f in (FIX / fixture).iterdir():
        if f.is_file():
            shutil.copy(f, dst / f.name)
    summary = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
    if summary_patch:
        for dotted, value in summary_patch.items():
            node = summary
            parts = dotted.split(".")
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = value
    store.finish_run(rid, "done", summary)
    store.set_score(rid, score)
    return rid


@pytest.fixture()
def env(tmp_path):
    """Store with a baseline + candidate val run and one linked official row."""
    store = Store(tmp_path)
    base_id = _mk_run(store, "val", "base", "base", score=dict(BASE_SCORE))
    cand_score = {"S": 0.82, "A": 0.9, "R": 0.95, "s_E": 0.5, "eta_ratio": 1.05,
                  "baseline_id": base_id, "weakest_class": None}
    cand_id = _mk_run(store, "val", "cand", "cand", score=dict(cand_score))
    store.set_baseline(base_id)
    oid = official.add(store, "val", "2026-10-20 09:00:00", "mock-base",
                       0.5, 0.9, 0.95, 0.4, 0.88, 0.99, 1.0, 300.0,
                       0.5, 43.0, 85, 100, 90.0, label="baseline v1",
                       local_run_id=base_id)
    return SimpleNamespace(store=store, base_id=base_id, cand_id=cand_id, oid=oid)


def _server(store):
    srv = dashboard.make_server(store, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th, srv.server_address[1]


def _get(port, path):
    url = f"http://127.0.0.1:{port}{path}"
    try:
        with urllib.request.urlopen(url) as resp:
            return resp.status, resp.headers, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


# ── build_state ───────────────────────────────────────────────────────────────
def test_state_keys_serialisable_and_calibration(env):
    st = dashboard.build_state(env.store, "val")
    for key in ("track", "generated_at", "progress", "baselines", "best", "history",
                "groups", "noise", "calibration", "charts"):
        assert key in st, f"missing key {key!r}"
    assert st["track"] == "val"
    assert st["progress"] is None  # no running run in the fixture store
    assert st["baselines"]["local"]["key"] == f"local:{env.base_id}"
    assert st["baselines"]["official"]["key"] == f"official:{env.oid}"
    assert st["baselines"]["local"]["is_baseline"] is True
    assert st["best"]["local"] is not None and st["best"]["official"] is not None
    assert set(st["charts"]) == {"s_over_time", "scatter", "eta_bars"}
    for svg in st["charts"].values():
        assert svg.startswith("<svg")
    # JSON-serialisable without allowing NaN/inf
    json.dumps(st, allow_nan=False)

    pairs = st["calibration"]["pairs"]
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["official_key"] == f"official:{env.oid}"
    assert pair["local_key"] == f"local:{env.base_id}"
    names = [m["name"] for m in pair["metrics"]]
    assert names == ["j_per_req", "slo_met_frac", "p95_ttft_s", "p95_tpot_ms",
                     "accuracy_pct", "eta_vs_base"]
    for m in pair["metrics"]:
        assert m["ratio"] is not None and m["flag"] in ("green", "amber", "red")


def test_state_unknown_track_raises(env):
    with pytest.raises(ValueError):
        dashboard.build_state(env.store, "banana")


def test_progress_running_fresh_and_stale(tmp_path):
    store = Store(tmp_path)
    rid = store.create_run("val", "live", "", {"model": "m"}, {})
    now = datetime.now()
    prog = {"items_done": 10, "items_total": 100,
            "started_at": (now - timedelta(minutes=1)).isoformat(),
            "updated_at": now.isoformat()}
    (store.run_dir(rid) / "progress.json").write_text(json.dumps(prog), encoding="utf-8")

    st = dashboard.build_state(store, "val")
    p = st["progress"]
    assert p is not None
    assert p["run_id"] == rid and p["track"] == "val" and p["label"] == "live"
    assert p["items_done"] == 10 and p["items_total"] == 100
    assert p["started_at"] == prog["started_at"]
    assert p["stale"] is False
    assert p["elapsed_s"] is not None and 50 <= p["elapsed_s"] <= 600
    json.dumps(st, allow_nan=False)

    prog["updated_at"] = (now - timedelta(minutes=5)).isoformat()
    (store.run_dir(rid) / "progress.json").write_text(json.dumps(prog), encoding="utf-8")
    st2 = dashboard.build_state(store, "val")
    assert st2["progress"]["stale"] is True


def test_progress_partial_json_tolerated(tmp_path):
    store = Store(tmp_path)
    rid = store.create_run("final", "partial", "", {"model": "m"}, {})
    (store.run_dir(rid) / "progress.json").write_text('{"items_done": 3', encoding="utf-8")
    st = dashboard.build_state(store, "final")
    p = st["progress"]
    assert p is not None and p["run_id"] == rid
    assert p["items_done"] is None and p["elapsed_s"] is None and p["stale"] is False
    json.dumps(st, allow_nan=False)


# ── build_run_detail ──────────────────────────────────────────────────────────
def test_run_detail_class_rows_flags_power(env):
    # a run whose rag class collapsed -> ratio 0.5/0.8 < 0.75 -> flag "low"
    bad_id = _mk_run(env.store, "val", "bad", "cand",
                     score={"S": 0.4, "A": 0.6, "R": 0.9, "s_E": 0.2,
                            "eta_ratio": 0.9, "baseline_id": env.base_id,
                            "weakest_class": "rag"},
                     summary_patch={"class_acc.rag": 0.5, "class_acc.long": 0.1})
    d = dashboard.build_run_detail(env.store, bad_id)
    assert d["row"]["key"] == f"local:{bad_id}"
    assert d["baseline_id"] == env.base_id
    assert d["report_url"] == f"/runs/{bad_id}/report.html"
    assert "<svg" in d["power_svg"]
    assert set(d["fail_by_phase"]) == {"ramp", "peak", "quiet", "drain"}
    assert isinstance(d["fail_by_phase"]["ramp"], dict)
    assert d["tokens_by_class"]["chat"]["mean"] is not None
    assert d["tokens_by_class"]["chat"]["p95"] is not None

    crs = {c["cls"]: c for c in d["class_rows"]}
    assert set(crs) == {"chat", "rag", "reason", "long", "thai", "agent"}
    rag, chat, long_ = crs["rag"], crs["chat"], crs["long"]
    assert rag["checked"] is True and rag["flag"] == "low"
    assert rag["base_acc"] == 0.8 and abs(rag["ratio"] - 0.625) < 1e-9
    assert chat["checked"] is True and chat["flag"] == "ok"
    assert long_["checked"] is False and long_["flag"] == "ok"
    json.dumps(d, allow_nan=False)

    # baseline run: ratio against itself, baseline_id = own id
    db = dashboard.build_run_detail(env.store, env.base_id)
    assert db["baseline_id"] == env.base_id
    for c in db["class_rows"]:
        assert c["ratio"] == pytest.approx(1.0) and c["flag"] == "ok"


def test_run_detail_without_summary(env):
    rid = env.store.create_run("val", "broken", "", {"model": "m"}, {})
    d = dashboard.build_run_detail(env.store, rid)
    assert d["row"]["key"] == f"local:{rid}"
    assert d["row"]["status"] == "running"
    assert d["summary"] is None and d["score"] is None
    assert d["class_rows"] == [] and d["fail_by_phase"] == {} and d["tokens_by_class"] == {}
    assert "<svg" in d["power_svg"]  # "no data" chart
    assert d["report_url"] == f"/runs/{rid}/report.html"
    json.dumps(d, allow_nan=False)


def test_run_detail_unknown_run_raises(env):
    with pytest.raises(ValueError):
        dashboard.build_run_detail(env.store, "no-such-run")


# ── HTTP ──────────────────────────────────────────────────────────────────────
def test_http_routes_headers_and_files(env):
    (env.store.run_dir(env.cand_id) / "report.html").write_text(
        "<html><body>tiny report</body></html>", encoding="utf-8")
    srv, th, port = _server(env.store)
    try:
        # '/'
        code, hdr, body = _get(port, "/")
        assert code == 200
        assert "text/html" in hdr["Content-Type"] and "charset=utf-8" in hdr["Content-Type"]
        assert hdr["Cache-Control"] == "no-store"
        assert int(hdr["Content-Length"]) == len(body)
        assert b"GreenMind local grader" in body

        # /api/state
        code, _, body = _get(port, "/api/state?track=val")
        assert code == 200
        data = json.loads(body)
        assert data["track"] == "val" and data["progress"] is None
        assert data["baselines"]["local"]["key"] == f"local:{env.base_id}"
        assert len(data["calibration"]["pairs"]) == 1

        # /api/runs -> local rows only
        code, _, body = _get(port, "/api/runs")
        assert code == 200
        rows = json.loads(body)
        assert all(r["source"] == "local" for r in rows)
        assert {r["id"] for r in rows} >= {env.base_id, env.cand_id}

        # /api/official -> official rows only
        code, _, body = _get(port, "/api/official")
        assert code == 200
        orows = json.loads(body)
        assert len(orows) == 1 and orows[0]["source"] == "official"

        # /api/run/<id>
        code, _, body = _get(port, f"/api/run/{env.cand_id}")
        assert code == 200
        detail = json.loads(body)
        assert detail["row"]["key"] == f"local:{env.cand_id}"
        assert detail["class_rows"] and detail["report_url"].endswith("report.html")

        # run files (whitelist)
        code, hdr, body = _get(port, f"/runs/{env.cand_id}/report.html")
        assert code == 200 and b"tiny report" in body
        assert "text/html" in hdr["Content-Type"] and "charset=utf-8" in hdr["Content-Type"]
        code, hdr, body = _get(port, f"/runs/{env.cand_id}/summary.json")
        assert code == 200 and json.loads(body)["requests"] == 140
        code, hdr, body = _get(port, f"/runs/{env.cand_id}/power.csv")
        assert code == 200 and body.startswith(b"t_s,watts,joules_cum")
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


def test_http_errors_and_traversal(env):
    srv, th, port = _server(env.store)
    try:
        # unknown track -> 400 JSON
        code, hdr, body = _get(port, "/api/state?track=banana")
        assert code == 400 and "application/json" in hdr["Content-Type"]
        assert "error" in json.loads(body)

        # unknown path -> 404 JSON
        code, _, body = _get(port, "/nope")
        assert code == 404 and "error" in json.loads(body)

        # unknown run -> 404
        code, _, body = _get(port, "/api/run/does-not-exist")
        assert code == 404 and "error" in json.loads(body)

        # file outside the whitelist -> 404
        code, _, body = _get(port, f"/runs/{env.cand_id}/records.jsonl")
        assert code == 404

        # path traversal attempts: 404/400, never file contents
        code, _, body = _get(port, "/runs/..%2f..%2fetc/passwd")
        assert code in (400, 404)
        assert b"root:" not in body and b"daemon:" not in body
        code, _, body = _get(port, f"/runs/{env.cand_id}/../../runs.sqlite")
        assert code in (400, 404)
        assert b"CREATE TABLE" not in body
        code, _, body = _get(port, f"/runs/..%2f{env.cand_id}%2fsummary.json")
        assert code in (400, 404)
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=5)


# ── export_static ─────────────────────────────────────────────────────────────
def test_export_static_embeds_data(env, tmp_path):
    out = tmp_path / "out" / "sub" / "dash.html"  # parent dirs must be created
    p = dashboard.export_static(env.store, out)
    assert p == out and p.is_file()
    html = p.read_text(encoding="utf-8")
    assert "http://" not in html

    m = re.search(r'<script id="export-data" type="application/json">(.*?)</script>',
                  html, re.S)
    assert m is not None, "export-data block missing"
    data = json.loads(m.group(1))
    assert set(data["states"]) == {"val", "final"}
    assert data["states"]["val"]["track"] == "val"
    assert data["states"]["final"]["track"] == "final"
    assert set(data["runs"]) == {env.base_id, env.cand_id}
    assert data["runs"][env.cand_id]["row"]["key"] == f"local:{env.cand_id}"
    # run ids are embedded in the html text
    assert env.base_id in html and env.cand_id in html
