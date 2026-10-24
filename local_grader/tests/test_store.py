"""Store tests (Agent C) — contract: INTERFACES.md."""
import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gm import store as store_mod
from gm.store import Store, format_list, format_show

REPO = Path(__file__).resolve().parent.parent
FIXTURE_SUMMARY = json.loads((REPO / "fixtures" / "base" / "summary.json").read_text(encoding="utf-8"))


class _FakeDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 1, 1, 12, 0, 12)


@pytest.fixture
def st(tmp_path):
    return Store(tmp_path)


# ── init / layout ─────────────────────────────────────────────────────────────
def test_init_creates_layout(tmp_path):
    s = Store(tmp_path)
    assert (tmp_path / "runs").is_dir()
    assert (tmp_path / "runs.sqlite").exists()
    assert s.list_runs() == []


# ── create / get / finish / list roundtrip ────────────────────────────────────
def test_create_get_finish_roundtrip(tmp_path):
    st = Store(tmp_path)
    rid = st.create_run("val", "smoke", "first run", {"model": "mock-model", "url": "http://127.0.0.1:9"}, {"TTFT_FROM": "any"})
    run = st.get_run(rid)
    assert run["id"] == rid
    assert run["status"] == "running"
    assert run["track"] == "val"
    assert run["label"] == "smoke"
    assert run["notes"] == "first run"
    assert run["is_baseline"] is False
    assert run["server"]["model"] == "mock-model"
    assert run["settings"]["TTFT_FROM"] == "any"
    assert run["summary"] is None
    assert run["score"] is None
    assert run["started_at"]
    assert run["ended_at"] is None

    summary = {"requests": 3, "on_time": 2, "eta": 0.01, "s_T": 0.5, "fail_reasons": {"ttft": 1}}
    st.finish_run(rid, "done", summary)
    got = st.get_run(rid)
    assert got["status"] == "done"
    assert got["ended_at"]
    assert got["summary"] == summary

    st.write_jsonl(rid, "records", [{"id": "r1", "cls": "chat"}])
    assert st.read_jsonl(rid, "records") == [{"id": "r1", "cls": "chat"}]
    st.write_json(rid, "summary", summary)
    assert st.read_json(rid, "summary") == summary

    rows = st.list_runs()
    assert len(rows) == 1 and rows[0]["id"] == rid and rows[0]["status"] == "done"


def test_list_newest_first_and_track_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "datetime", _FakeDT)
    st = Store(tmp_path)
    st.create_run("val", "alpha", "", {"model": "m"}, {})
    st.create_run("final", "beta", "", {"model": "m"}, {})
    st.create_run("val", "gamma", "", {"model": "m"}, {})
    assert [r["id"] for r in st.list_runs()] == [
        "20260101-120012-gamma", "20260101-120012-beta", "20260101-120012-alpha"]
    assert [r["id"] for r in st.list_runs("val")] == ["20260101-120012-gamma", "20260101-120012-alpha"]
    assert [r["id"] for r in st.list_runs(limit=2)] == ["20260101-120012-gamma", "20260101-120012-beta"]


# ── id slug + clash suffix ────────────────────────────────────────────────────
def test_id_slug_and_clash(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "datetime", _FakeDT)
    st = Store(tmp_path)
    rid = st.create_run("val", "Smoke Test!", "", {"model": "m"}, {})
    assert rid == "20260101-120012-smoke-test-"
    rid_empty = st.create_run("val", "", "", {"model": "m"}, {})
    assert rid_empty == "20260101-120012-run"
    rid2 = st.create_run("val", "Smoke Test!", "", {"model": "m"}, {})
    assert rid2 == "20260101-120012-smoke-test--2"


# ── baselines ─────────────────────────────────────────────────────────────────
def test_set_baseline_requires_done(tmp_path):
    st = Store(tmp_path)
    rid = st.create_run("val", "x", "", {"model": "m"}, {})
    with pytest.raises(ValueError):
        st.set_baseline(rid)
    st.finish_run(rid, "failed", None)
    with pytest.raises(ValueError):
        st.set_baseline(rid)


def test_exactly_one_baseline_per_track(tmp_path):
    st = Store(tmp_path)
    r1 = st.create_run("val", "b1", "", {"model": "m"}, {})
    st.finish_run(r1, "done", {"eta": 1.0})
    r2 = st.create_run("val", "b2", "", {"model": "m"}, {})
    st.finish_run(r2, "done", {"eta": 2.0})
    st.set_baseline(r1)
    assert st.get_baseline("val")["id"] == r1
    st.set_baseline(r2)
    assert st.get_baseline("val")["id"] == r2
    assert st.get_run(r1)["is_baseline"] is False
    assert st.get_run(r2)["is_baseline"] is True
    # a different track keeps its own baseline
    f1 = st.create_run("final", "f1", "", {"model": "m"}, {})
    st.finish_run(f1, "done", {"eta": 1.0})
    st.set_baseline(f1)
    assert st.get_baseline("final")["id"] == f1
    assert st.get_baseline("val")["id"] == r2


# ── resolve ────────────────────────────────────────────────────────────────────
def test_resolve_exact_prefix_label_ambiguous_no_run(tmp_path, monkeypatch):
    monkeypatch.setattr(store_mod, "datetime", _FakeDT)
    st = Store(tmp_path)
    r1 = st.create_run("val", "alpha", "", {"model": "m"}, {})
    r2 = st.create_run("val", "beta", "", {"model": "m"}, {})
    g1 = st.create_run("val", "gamma", "", {"model": "m"}, {})
    g2 = st.create_run("val", "gamma", "", {"model": "m"}, {})
    assert r1 == "20260101-120012-alpha"
    assert r2 == "20260101-120012-beta"
    assert st.resolve(r1) == r1
    assert st.resolve("20260101-120012-al") == r1                      # unique prefix
    assert st.resolve("alpha") == r1                                  # unique label
    with pytest.raises(ValueError, match="ambiguous"):
        st.resolve("20260101-120012")                                   # prefix of several
    with pytest.raises(ValueError, match="ambiguous"):
        st.resolve("gamma")                                             # two runs, same label
    with pytest.raises(ValueError, match="no run"):
        st.resolve("does-not-exist")


# ── path traversal ────────────────────────────────────────────────────────────
def test_run_dir_rejects_traversal(st):
    with pytest.raises(ValueError):
        st.run_dir("../evil")
    with pytest.raises(ValueError):
        st.run_dir("a/b")
    with pytest.raises(ValueError):
        st.run_dir("..")


# ── format_list ───────────────────────────────────────────────────────────────
def test_format_list_empty():
    assert format_list([]) == "no runs"


def test_format_list_fixture_run(st):
    rid = st.create_run("val", "fixture", "", {"model": "mock-model"}, {})
    st.finish_run(rid, "done", FIXTURE_SUMMARY)
    st.set_score(rid, {"S": 55.5, "A": 0.9, "s_E": 0.4, "eta_ratio": 1.2, "weakest_class": ["thai", 0.625]})
    run = st.get_run(rid)
    text = format_list([run])
    assert run["id"] in text
    assert "55.5" in text          # S, 1 decimal
    assert "129/142" in text       # on-time
    assert "0.00369" in text       # eta, 5 decimals
    assert format_list([]) == "no runs"


# ── format_show ───────────────────────────────────────────────────────────────
def test_format_show_fixture_run(st):
    rid = st.create_run("val", "fixture", "", {"model": "mock-model"}, {})
    st.finish_run(rid, "done", FIXTURE_SUMMARY)
    st.set_score(rid, {"S": 55.5, "A": 0.9, "s_E": 0.4, "eta_ratio": 1.2, "weakest_class": ["thai", 0.625]})
    text = format_show(st.get_run(rid))
    assert f"id: {rid}" in text
    assert "model: mock-model" in text
    assert "on-time: 129/142" in text
    assert "energy_J: 34958.013" in text
    assert "eta: 0.00369" in text
    assert "s_T: 0.8833" in text
    assert "score:" in text
    assert "S: 55.5" in text


def test_format_show_run_without_summary_never_raises(st):
    rid = st.create_run("val", "bare", "", {"model": "mock-model"}, {})
    st.finish_run(rid, "interrupted", None)
    text = format_show(st.get_run(rid))
    assert "id: " in text
    assert "on-time: -" in text
    assert "score:" in text
    assert format_list([st.get_run(rid)])  # S n/a / on-time - / eta - must not raise
