"""gm/official tests (Agent O) — contract: INTERFACES2.md §2."""
import os
import sqlite3

import pytest

from gm import official
from gm.store import Store

HEADER = ["ID", "TRACK", "STARTED", "LABEL", "S", "A", "R", "sE", "sT", "sL",
          "ETA", "J/REQ", "P95TTFT", "P95TPOT", "SLO(met/total)", "ACC%", "LINK"]

# the 13 metric arguments S..accuracy_pct (all None)
NONE_METRICS = (None,) * 13


@pytest.fixture
def st(tmp_path):
    return Store(tmp_path)


def _add(st, track="val", started="2026-10-24 04:50:00", label="", local_run_id=None, **kw):
    """Seed-row values by default; kw overrides individual metrics."""
    seed = dict(model="mock-model", S=0.0, A=1.0, R=1.0, s_E=0.0, s_T=0.51, s_L=0.59,
                eta_vs_base=0.89, j_per_req=467, p95_ttft_s=2.06, p95_tpot_ms=48,
                slo_met=64, slo_total=100, accuracy_pct=87.2)
    seed.update(kw)
    return official.add(st, track, started, label=label, local_run_id=local_run_id, **seed)


# ── add / get / list round trip (exact seed values) ───────────────────────────
def test_add_get_list_roundtrip_seed_values(st):
    oid = official.add(st, "val", "2026-10-24 04:50:00", "mock-model",
                       0.0, 1.0, 1.0, 0.0, 0.51, 0.59, 0.89, 467, 2.06, 48, 64, 100, 87.2,
                       label="baseline", notes="seed row")
    assert isinstance(oid, int) and oid == 1

    row = official.get(st, oid)
    assert set(row) == set(official.COLUMNS)
    assert row["id"] == 1
    assert row["track"] == "val"
    assert row["started_at"] == "2026-10-24T04:50:00"
    assert row["model"] == "mock-model"
    assert row["label"] == "baseline"
    assert row["notes"] == "seed row"
    assert row["S"] == 0.0
    assert row["A"] == 1.0
    assert row["R"] == 1.0
    assert row["s_E"] == 0.0
    assert row["s_T"] == 0.51
    assert row["s_L"] == 0.59
    assert row["eta_vs_base"] == 0.89
    assert row["j_per_req"] == 467.0
    assert row["p95_ttft_s"] == 2.06
    assert row["p95_tpot_ms"] == 48.0
    assert row["slo_met"] == 64
    assert row["slo_total"] == 100
    assert row["accuracy_pct"] == 87.2
    assert row["local_run_id"] is None
    assert row["created_at"]

    rows = official.list_official(st)
    assert len(rows) == 1
    assert rows[0] == row
    assert isinstance(rows[0], dict) and set(rows[0]) == set(official.COLUMNS)


def test_add_metrics_as_strings_and_none(st):
    oid = official.add(st, "val", "2026-10-24 06:00:00", "m",
                       "0.0", "1.0", "1.0", "0.0", "0.5", "0.6", "0.9",
                       "400", "2.0", "47", "60", "100", "85.0")
    row = official.get(st, oid)
    assert row["S"] == 0.0 and row["A"] == 1.0 and row["j_per_req"] == 400.0
    assert row["p95_tpot_ms"] == 47.0
    assert row["slo_met"] == 60 and row["slo_total"] == 100
    assert row["accuracy_pct"] == 85.0
    oid2 = official.add(st, "final", "2026-10-24 07:00:00", "m", *NONE_METRICS)
    row2 = official.get(st, oid2)
    assert row2["S"] is None and row2["A"] is None and row2["slo_met"] is None
    assert row2["slo_total"] is None and row2["accuracy_pct"] is None


# ── started normalisation ─────────────────────────────────────────────────────
def test_started_normalisation(st):
    o1 = official.add(st, "val", "2026-10-24 04:50:00", "m", *NONE_METRICS)
    o2 = official.add(st, "val", "2026-10-24T04:50:00", "m", *NONE_METRICS)
    assert official.get(st, o1)["started_at"] == "2026-10-24T04:50:00"
    assert official.get(st, o2)["started_at"] == "2026-10-24T04:50:00"


def test_bad_track_and_date_raise(st):
    with pytest.raises(ValueError):
        official.add(st, "dev", "2026-10-24 04:50:00", "m", *NONE_METRICS)
    with pytest.raises(ValueError):
        official.add(st, "VAL", "2026-10-24 04:50:00", "m", *NONE_METRICS)
    for bad in ("2026-13-99 00:00:00", "24 October 2026", "garbage", None, ""):
        with pytest.raises(ValueError):
            official.add(st, "val", bad, "m", *NONE_METRICS)


# ── get / list / delete errors & behaviour ────────────────────────────────────
def test_get_missing_raises(st):
    with pytest.raises(ValueError):
        official.get(st, 9999)


def test_list_newest_first_and_track_filter(st):
    a = _add(st, started="2026-10-24 04:00:00")            # val, oldest
    b = _add(st, started="2026-10-25 04:00:00")            # val
    c = _add(st, "final", started="2026-10-24 12:00:00")   # final
    d = _add(st, started="2026-10-25 04:00:00")            # val, same start as b, newer id
    rows = official.list_official(st)
    assert [r["id"] for r in rows] == [d, b, c, a]
    assert [r["id"] for r in official.list_official(st, "val")] == [d, b, a]
    assert [r["id"] for r in official.list_official(st, "final")] == [c]
    assert official.list_official(st, "nope") == []


# ── link ───────────────────────────────────────────────────────────────────────
def test_link_success_and_relink_replaces(st):
    oid = _add(st)
    r1 = st.create_run("val", "x", "", {}, {})
    st.finish_run(r1, "done", {})
    r2 = st.create_run("val", "y", "", {}, {})
    st.finish_run(r2, "done", {})
    assert official.link(st, oid, r1) == r1
    assert official.get(st, oid)["local_run_id"] == r1
    assert official.link(st, oid, r2) == r2          # re-link replaces
    assert official.get(st, oid)["local_run_id"] == r2


def test_link_bad_track_and_unknown_run_raise(st):
    oid = _add(st)
    rv = st.create_run("val", "x", "", {}, {})
    st.finish_run(rv, "done", {})
    rf = st.create_run("final", "f", "", {}, {})
    st.finish_run(rf, "done", {})
    with pytest.raises(ValueError):
        official.link(st, oid, rf)                   # other track
    with pytest.raises(ValueError):
        official.link(st, oid, "does-not-exist")     # unknown run
    # an ambiguous prefix (shared by both runs, labels differ) also raises
    with pytest.raises(ValueError):
        official.link(st, oid, os.path.commonprefix([rv, rf]))
    assert official.get(st, oid)["local_run_id"] is None  # failed links change nothing


def test_link_missing_official_raises(st):
    rv = st.create_run("val", "x", "", {}, {})
    st.finish_run(rv, "done", {})
    with pytest.raises(ValueError):
        official.link(st, 42, rv)


# ── add with local_run_id ─────────────────────────────────────────────────────
def test_add_with_local_run_id(st):
    rv = st.create_run("val", "x", "", {}, {})
    st.finish_run(rv, "done", {})
    oid = _add(st, local_run_id=rv)
    assert official.get(st, oid)["local_run_id"] == rv
    rf = st.create_run("final", "f", "", {}, {})
    st.finish_run(rf, "done", {})
    with pytest.raises(ValueError):
        _add(st, local_run_id=rf)                    # run on the other track
    with pytest.raises(ValueError):
        _add(st, local_run_id="no-such-run")         # unknown run
    with pytest.raises(ValueError):
        _add(st, local_run_id=os.path.commonprefix([rv, rf]))  # ambiguous prefix


def test_add_without_local_run_id_keeps_none(st):
    oid = _add(st)
    assert official.get(st, oid)["local_run_id"] is None


# ── delete ─────────────────────────────────────────────────────────────────────
def test_delete(st):
    o1 = _add(st, started="2026-10-24 04:00:00")
    o2 = _add(st, started="2026-10-24 05:00:00")
    official.delete(st, o1)
    rows = official.list_official(st)
    assert [r["id"] for r in rows] == [o2]
    with pytest.raises(ValueError):
        official.get(st, o1)
    official.delete(st, o1)                          # deleting twice is a no-op


# ── same sqlite file as the store ─────────────────────────────────────────────
def test_table_lives_in_store_db_file(st):
    oid = _add(st)
    conn = sqlite3.connect(str(st.db_path))
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        cols = [r[1] for r in conn.execute("PRAGMA table_info(official)")]
        pk = [r for r in conn.execute("PRAGMA table_info(official)") if r[5] > 0]
    finally:
        conn.close()
    assert "runs" in names and "official" in names
    assert cols == official.COLUMNS
    assert [p[1] for p in pk] == ["id"]
    assert official.get(st, oid)["id"] == oid        # same file, same data


def test_init_table_idempotent(st):
    official.init_table(st)
    official.init_table(st)                           # IF NOT EXISTS
    oid = _add(st)
    assert official.get(st, oid)["id"] == oid


# ── format_official_list ──────────────────────────────────────────────────────
def test_format_official_list_empty():
    assert official.format_official_list([]) == "no official results"
    assert official.format_official_list(None) == "no official results"


def test_format_official_list_rows(st):
    rv = st.create_run("val", "x", "", {}, {})
    st.finish_run(rv, "done", {})
    oid = _add(st, label="baseline", local_run_id=rv)
    # a second row with all-None metrics and a later start (so it sorts first)
    oid2 = _add(st, started="2026-10-24 05:00:00", **{k: None for k in
                  ("model", "S", "A", "R", "s_E", "s_T", "s_L", "eta_vs_base",
                   "j_per_req", "p95_ttft_s", "p95_tpot_ms", "slo_met", "slo_total",
                   "accuracy_pct")})
    text = official.format_official_list(official.list_official(st))
    lines = text.splitlines()
    assert len(lines) == 3
    assert lines[0].split() == HEADER

    def col_starts(line):
        starts, i = [], 0
        while i < len(line):
            if line[i] != " ":
                starts.append(i)
                while i < len(line) and line[i] != " ":
                    i += 1
            else:
                i += 1
        return starts

    header_starts = col_starts(lines[0])
    assert len(header_starts) == 17
    for l in lines:
        assert len(l.split()) == 17
        assert col_starts(l.ljust(len(lines[0]))) == header_starts  # fixed-width columns

    c2 = lines[1].split()                                       # newest first = oid2 (all None → '-')
    assert c2[0] == str(oid2)
    assert c2[1] == "val"
    assert c2[2] == "2026-10-24T05:00:00"
    assert c2[4:] == ["-"] * 13
    c1 = lines[2].split()                                       # seed row, linked
    assert c1 == [str(oid), "val", "2026-10-24T04:50:00", "baseline",
                  "0.0", "1.00", "1.000", "0.00", "0.51", "0.59", "0.89",
                  "467", "2.06", "48", "64/100", "87.2", rv]
