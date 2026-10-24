"""Official results log: SQLite table `official` inside the SAME file as the store.

Contract: INTERFACES2.md §2 (owner: Agent O). Stdlib only.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime

COLUMNS = ["id", "track", "started_at", "model", "label", "notes", "S", "A", "R",
           "s_E", "s_T", "s_L", "eta_vs_base", "j_per_req", "p95_ttft_s",
           "p95_tpot_ms", "slo_met", "slo_total", "accuracy_pct", "local_run_id", "created_at"]

_TRACKS = ("val", "final")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS official (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track TEXT,
    started_at TEXT,
    model TEXT,
    label TEXT,
    notes TEXT,
    S REAL,
    A REAL,
    R REAL,
    s_E REAL,
    s_T REAL,
    s_L REAL,
    eta_vs_base REAL,
    j_per_req REAL,
    p95_ttft_s REAL,
    p95_tpot_ms REAL,
    slo_met INTEGER,
    slo_total INTEGER,
    accuracy_pct REAL,
    local_run_id TEXT,
    created_at TEXT
)
"""

# (header, row key, decimals | None for text)
_HEADER = [
    ("ID", "id", None), ("TRACK", "track", None), ("STARTED", "started_at", None),
    ("LABEL", "label", None),
    ("S", "S", 1), ("A", "A", 2), ("R", "R", 3),
    ("sE", "s_E", 2), ("sT", "s_T", 2), ("sL", "s_L", 2),
    ("ETA", "eta_vs_base", 2), ("J/REQ", "j_per_req", 0),
    ("P95TTFT", "p95_ttft_s", 2), ("P95TPOT", "p95_tpot_ms", 0),
    ("SLO(met/total)", None, None), ("ACC%", "accuracy_pct", 1),
    ("LINK", "local_run_id", None),
]


@contextmanager
def _conn(store):
    """One sqlite connection per call; WAL on; Row factory; commit/close on exit.

    Mirrors gm.store.Store._conn but operates on store.db_path directly.
    """
    conn = sqlite3.connect(str(store.db_path), timeout=30.0)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _as_float(value):
    """float/str → float; None → None; anything else → ValueError."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"invalid number {value!r}")


def _as_int(value):
    """float/str → int; None → None; anything else → ValueError."""
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise ValueError(f"invalid number {value!r}")


def _norm_started(value):
    """'YYYY-MM-DD HH:MM:SS' or ISO (with T) → 'YYYY-MM-DDTHH:MM:SS'; invalid → ValueError."""
    if value is None:
        raise ValueError("started is required")
    text = str(value).strip()
    if not text:
        raise ValueError("started is required")
    text = text.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"invalid started {value!r}") from exc
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def init_table(store) -> None:
    """CREATE TABLE IF NOT EXISTS official in the store's sqlite file (idempotent)."""
    with _conn(store) as conn:
        conn.execute(_CREATE_TABLE)


def _row(row) -> dict:
    """sqlite Row → plain dict with exactly the COLUMNS keys."""
    return {k: row[k] for k in COLUMNS}


def add(store, track, started, model, S, A, R, s_E, s_T, s_L, eta_vs_base, j_per_req,
        p95_ttft_s, p95_tpot_ms, slo_met, slo_total, accuracy_pct, label="", notes="",
        local_run_id=None) -> int:
    """Insert one official row; returns the new id (int).

    - track must be in {val, final} (ValueError otherwise).
    - started accepts 'YYYY-MM-DD HH:MM:SS' or ISO with T; stored as ISO 'YYYY-MM-DDTHH:MM:SS'.
    - metrics may be float or str (converted; None allowed).
    - local_run_id (id/prefix/label) is resolved via store.resolve and must be on the
      same track as this row (ValueError otherwise).
    """
    init_table(store)
    track = str(track)
    if track not in _TRACKS:
        raise ValueError(f"track must be one of {list(_TRACKS)}, got {track!r}")
    started_iso = _norm_started(started)
    values = {
        "S": _as_float(S), "A": _as_float(A), "R": _as_float(R),
        "s_E": _as_float(s_E), "s_T": _as_float(s_T), "s_L": _as_float(s_L),
        "eta_vs_base": _as_float(eta_vs_base), "j_per_req": _as_float(j_per_req),
        "p95_ttft_s": _as_float(p95_ttft_s), "p95_tpot_ms": _as_float(p95_tpot_ms),
        "accuracy_pct": _as_float(accuracy_pct),
        "slo_met": _as_int(slo_met), "slo_total": _as_int(slo_total),
    }
    resolved_run_id = None
    if local_run_id is not None:
        run_id = store.resolve(local_run_id)
        run = store.get_run(run_id)
        if run["track"] != track:
            raise ValueError(f"run {run_id} is on track {run['track']!r},"
                             f" official row is on {track!r}")
        resolved_run_id = run_id
    with _conn(store) as conn:
        cur = conn.execute(
            "INSERT INTO official (track, started_at, model, label, notes, S, A, R, s_E, s_T, s_L,"
            " eta_vs_base, j_per_req, p95_ttft_s, p95_tpot_ms, slo_met, slo_total, accuracy_pct,"
            " local_run_id, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (track, started_iso, model, label, notes,
             values["S"], values["A"], values["R"], values["s_E"], values["s_T"], values["s_L"],
             values["eta_vs_base"], values["j_per_req"], values["p95_ttft_s"], values["p95_tpot_ms"],
             values["slo_met"], values["slo_total"], values["accuracy_pct"],
             resolved_run_id, datetime.now().isoformat()))
        return int(cur.lastrowid)


def get(store, official_id) -> dict:
    """Full row as a plain dict (COLUMNS keys); ValueError if missing."""
    init_table(store)
    with _conn(store) as conn:
        row = conn.execute("SELECT * FROM official WHERE id = ?", (int(official_id),)).fetchone()
    if row is None:
        raise ValueError(f"no official result {official_id}")
    return _row(row)


def list_official(store, track=None) -> list:
    """Rows as plain dicts (COLUMNS keys), newest first by started_at then id."""
    init_table(store)
    sql = "SELECT * FROM official"
    params = []
    if track is not None:
        sql += " WHERE track = ?"
        params.append(str(track))
    sql += " ORDER BY started_at DESC, id DESC"
    with _conn(store) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row(r) for r in rows]


def link(store, official_id, run_id) -> str:
    """Set the official row's local_run_id; returns the full run id.

    The official row must exist; the run is resolved with store.resolve and must be
    on the same track (ValueError otherwise). Replaces any previous link.
    """
    init_table(store)
    orow = get(store, official_id)
    rid = store.resolve(run_id)
    run = store.get_run(rid)
    if run["track"] != orow["track"]:
        raise ValueError(f"run {rid} is on track {run['track']!r},"
                         f" official row {official_id} is on {orow['track']!r}")
    with _conn(store) as conn:
        conn.execute("UPDATE official SET local_run_id = ? WHERE id = ?", (rid, int(official_id)))
    return rid


def delete(store, official_id) -> None:
    """Delete the official row (no-op when it does not exist)."""
    init_table(store)
    with _conn(store) as conn:
        conn.execute("DELETE FROM official WHERE id = ?", (int(official_id),))


def _num(value, nd: int) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _txt(value) -> str:
    if value is None:
        return "-"
    s = str(value)
    return s if s.strip() != "" else "-"


def format_official_list(rows) -> str:
    """Fixed-width table: header + one line per row; '-' for None; empty → 'no official results'."""
    rows = list(rows or [])
    if not rows:
        return "no official results"

    def cells(r) -> list:
        r = r if isinstance(r, dict) else {}
        met, total = r.get("slo_met"), r.get("slo_total")
        slo = f"{met}/{total}" if met is not None and total is not None else "-"
        out = []
        for _header, key, nd in _HEADER:
            if key is None:
                out.append(slo)
            elif nd is None:
                out.append(_txt(r.get(key)))
            else:
                out.append(_num(r.get(key), nd))
        return out

    all_cells = [cells(r) for r in rows]
    widths = [len(h) for h, _k, _nd in _HEADER]
    for c in all_cells:
        for i, cell in enumerate(c):
            widths[i] = max(widths[i], len(cell))
    lines = [" ".join(h.ljust(w) for (h, _k, _nd), w in zip(_HEADER, widths)).rstrip()]
    for c in all_cells:
        lines.append(" ".join(cell.ljust(w) for cell, w in zip(c, widths)).rstrip())
    return "\n".join(lines)
