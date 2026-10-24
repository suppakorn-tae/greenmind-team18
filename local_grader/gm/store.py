"""Run store: sqlite registry (root/runs.sqlite) + json sidecar files (root/runs/<run_id>/).
Contract: INTERFACES.md. Owner: Agent C."""
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

_GRADER_ROOT = Path(__file__).resolve().parent.parent

_STATUSES = ("done", "interrupted", "failed")

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    track TEXT,
    label TEXT,
    notes TEXT,
    status TEXT,
    started_at TEXT,
    ended_at TEXT,
    is_baseline INTEGER,
    server_json TEXT,
    settings_json TEXT,
    summary_json TEXT,
    score_json TEXT
)
"""

_SLUG_BAD = re.compile(r"[^a-z0-9._-]")


def _slug(label) -> str:
    """Lowercase, non [a-z0-9._-] → '-', max 40 chars, empty → 'run'."""
    slug = _SLUG_BAD.sub("-", str(label if label is not None else "").lower())[:40]
    return slug or "run"


def _now_iso() -> str:
    return datetime.now().isoformat()


def _jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _jload(text):
    """JSON text → dict (or None when null / unreadable)."""
    if text is None:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


class Store:
    """Local LLM-grader run registry."""

    def __init__(self, root=None):
        if root is None:
            root = os.environ.get("GRADER_HOME") or str(_GRADER_ROOT)
        self.root = Path(root)
        self.runs_dir = self.root / "runs"
        self.db_path = self.root / "runs.sqlite"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(_CREATE_RUNS)

    # ── plumbing ──────────────────────────────────────────────────────────────
    @contextmanager
    def _conn(self):
        """One sqlite connection per call; closed + committed on exit. WAL on. Thread-safe."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
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

    @staticmethod
    def _decode(row) -> dict:
        return {
            "id": row["id"],
            "track": row["track"],
            "label": row["label"],
            "notes": row["notes"],
            "status": row["status"],
            "started_at": row["started_at"],
            "ended_at": row["ended_at"],
            "is_baseline": bool(row["is_baseline"]),
            "server": _jload(row["server_json"]),
            "settings": _jload(row["settings_json"]),
            "summary": _jload(row["summary_json"]),
            "score": _jload(row["score_json"]),
        }

    # ── runs ───────────────────────────────────────────────────────────────────
    def create_run(self, track, label, notes, server, settings) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{stamp}-{_slug(label)}"
        run_id = base
        n = 1
        while True:
            with self._conn() as conn:
                clash = conn.execute("SELECT 1 FROM runs WHERE id = ?", (run_id,)).fetchone() is not None
            if not clash:
                break
            n += 1
            run_id = f"{base}-{n}"
        self.run_dir(run_id).mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, track, label, notes, status, started_at, ended_at, is_baseline,"
                " server_json, settings_json, summary_json, score_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, track, label, notes, "running", _now_iso(), None, 0,
                 _jdump(server) if server is not None else None,
                 _jdump(settings) if settings is not None else None,
                 None, None))
        return run_id

    def run_dir(self, run_id) -> Path:
        rid = str(run_id)
        if not rid or any(ch in rid for ch in ("/", "\\")) or ".." in rid.split("/"):
            raise ValueError(f"invalid run id: {run_id!r}")
        return self.root / "runs" / rid

    def finish_run(self, run_id, status, summary: dict | None) -> None:
        if status not in _STATUSES:
            raise ValueError(f"invalid status {status!r}")
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, ended_at = ?, summary_json = ? WHERE id = ?",
                (status, _now_iso(), _jdump(summary) if summary is not None else None, str(run_id)))

    def set_score(self, run_id, score: dict | None) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE runs SET score_json = ? WHERE id = ?",
                         (_jdump(score) if score is not None else None, str(run_id)))

    def update_run(self, run_id, settings: dict | None = None, summary: dict | None = None) -> None:
        """Rescore write-back: swap stored settings / summary for one run."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET settings_json = ?, summary_json = ? WHERE id = ?",
                (_jdump(settings) if settings is not None else None,
                 _jdump(summary) if summary is not None else None, str(run_id)))

    def get_run(self, run_id) -> dict:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (str(run_id),)).fetchone()
        if row is None:
            raise ValueError(f"no run {run_id}")
        return self._decode(row)

    def list_runs(self, track=None, limit: int = 20) -> list:
        sql = "SELECT * FROM runs"
        params = []
        if track is not None:
            sql += " WHERE track = ?"
            params.append(track)
        sql += " ORDER BY started_at DESC, id DESC LIMIT ?"
        params.append(int(limit))
        out = []
        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        for row in rows:
            out.append(self._decode(row))
        return out

    # ── baselines ─────────────────────────────────────────────────────────────
    def set_baseline(self, run_id) -> None:
        run = self.get_run(run_id)
        if run["status"] != "done":
            raise ValueError(f"run {run_id} status is {run['status']!r}; cannot set baseline")
        with self._conn() as conn:
            conn.execute("UPDATE runs SET is_baseline = 0 WHERE track = ? AND is_baseline = 1", (run["track"],))
            conn.execute("UPDATE runs SET is_baseline = 1 WHERE id = ?", (str(run_id),))

    def get_baseline(self, track) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE track = ? AND is_baseline = 1", (track,)).fetchone()
        return self._decode(row) if row is not None else None

    # ── refs ───────────────────────────────────────────────────────────────────
    def resolve(self, ref) -> str:
        ref = str(ref)
        with self._conn() as conn:
            rows = conn.execute("SELECT id, label FROM runs").fetchall()
        ids = [r["id"] for r in rows]
        if ref in ids:
            return ref
        matches = sorted(i for i in ids if i.startswith(ref))
        if len(matches) > 1:
            raise ValueError(f"ambiguous {ref}")
        if len(matches) == 1:
            return matches[0]
        labels = sorted(r["id"] for r in rows if str(r["label"] or "").lower() == ref.lower())
        if len(labels) > 1:
            raise ValueError(f"ambiguous {ref}")
        if len(labels) == 1:
            return labels[0]
        raise ValueError(f"no run {ref}")

    # ── json sidecars ──────────────────────────────────────────────────────────
    @staticmethod
    def _fname(name, ext) -> str:
        """`records` and `records.jsonl` both name the same file."""
        return name if name.endswith(ext) else f"{name}{ext}"

    def write_jsonl(self, run_id, name, rows) -> None:
        d = self.run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        text = "".join(_jdump(r) + "\n" for r in (rows or []))
        (d / self._fname(name, ".jsonl")).write_text(text, encoding="utf-8")

    def read_jsonl(self, run_id, name) -> list:
        p = self.run_dir(run_id) / self._fname(name, ".jsonl")
        if not p.exists():
            return []
        out = []
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out

    def write_json(self, run_id, name, obj) -> None:
        d = self.run_dir(run_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / self._fname(name, ".json")).write_text(_jdump(obj), encoding="utf-8")

    def read_json(self, run_id, name):
        p = self.run_dir(run_id) / self._fname(name, ".json")
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))


# ── formatting ─────────────────────────────────────────────────────────────────
def _n(v, nd: int = 2) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def _as_dict(v) -> dict:
    return v if isinstance(v, dict) else {}


def format_list(rows) -> str:
    rows = list(rows or [])
    if not rows:
        return "no runs"
    header = f"{'ID':<20} {'TRACK':<8} {'LABEL':<14} {'STATUS':<10} {'BASE':<6} {'S':>6} {'ON-TIME':>12} {'ETA':>10}"
    lines = [header]
    for r in rows:
        r = _as_dict(r)
        base = "*" if r.get("is_baseline") else "-"
        score = _as_dict(r.get("score"))
        s = score.get("S")
        s_txt = f"{float(s):.1f}" if isinstance(s, (int, float)) and not isinstance(s, bool) else "n/a"
        summary = _as_dict(r.get("summary"))
        on_time, requests = summary.get("on_time"), summary.get("requests")
        on_txt = f"{on_time}/{requests}" if on_time is not None and requests is not None else "-"
        eta = summary.get("eta")
        eta_txt = f"{float(eta):.5f}" if eta is not None else "-"
        lines.append(
            f"{str(r.get('id') or ''):<20} {str(r.get('track') or ''):<8} {str(r.get('label') or ''):<14} "
            f"{str(r.get('status') or ''):<10} {base:<6} {s_txt:>6} {on_txt:>12} {eta_txt:>10}")
    return "\n".join(lines)


def format_show(run) -> str:
    r = _as_dict(run)
    lines = []

    def add(key, value):
        lines.append(f"{key}: {value}")

    def txt(v) -> str:
        if v is None:
            return "-"
        s = str(v)
        return "-" if s.strip() == "" else s

    add("id", txt(r.get("id")))
    add("track", txt(r.get("track")))
    add("label", txt(r.get("label")))
    add("notes", txt(r.get("notes")))
    add("status", txt(r.get("status")))
    add("started_at", txt(r.get("started_at")))
    add("ended_at", txt(r.get("ended_at")))
    add("baseline", txt(r.get("is_baseline")))
    server = _as_dict(r.get("server"))
    add("model", txt(server.get("model")))

    summary = _as_dict(r.get("summary"))
    on_time, requests = summary.get("on_time"), summary.get("requests")
    add("on-time", f"{on_time}/{requests}" if on_time is not None and requests is not None else "-")
    fails = summary.get("fail_reasons")
    add("fail_reasons", json.dumps(fails, ensure_ascii=False, sort_keys=True) if fails else "-")
    add("energy_J", _n(summary.get("energy_J"), 3))
    add("avg_power_W", _n(summary.get("avg_power_W"), 3))
    add("idle_W", _n(summary.get("idle_W"), 3))
    add("eta", _n(summary.get("eta"), 5))
    add("tokens_per_J", _n(summary.get("tokens_per_J"), 3))
    add("s_T", _n(summary.get("s_T"), 4))
    add("s_L", _n(summary.get("s_L"), 4))
    add("p95_tpot_ms", _n(summary.get("p95_tpot_ms"), 3))
    add("p95_ttft_over_slo", _n(summary.get("p95_ttft_over_slo"), 4))
    add("macro_acc", _n(summary.get("macro_acc"), 4))
    st_, sa_ = _as_dict(summary.get("slo_turn")), _as_dict(summary.get("slo_arrival"))
    if sa_:
        add("slo_met (per arrival)", f"{sa_.get('on_time', '-')}/{sa_.get('total', '-')}")
        add("slo_met (per turn)", f"{st_.get('on_time', '-')}/{st_.get('total', '-')}")
    add("j_per_req", _n(summary.get("j_per_req"), 1))
    add("p95_ttft_s", _n(summary.get("p95_ttft_s"), 3))
    add("accuracy_pct", _n(summary.get("accuracy_pct"), 1))

    class_acc = summary.get("class_acc")
    class_n = summary.get("class_n")
    if isinstance(class_acc, dict) and class_acc:
        for cls in sorted(class_acc):
            n = class_n.get(cls) if isinstance(class_n, dict) else None
            lines.append(f"class_acc {cls}: acc={_n(class_acc[cls], 4)} n={txt(n)}")
    else:
        lines.append("class_acc: -")

    score = _as_dict(r.get("score"))
    s, a = score.get("S"), score.get("A")
    s_txt = f"{float(s):.1f}" if isinstance(s, (int, float)) and not isinstance(s, bool) else "n/a"
    weakest = score.get("weakest_class")
    weakest_txt = f"{weakest[0]}:{weakest[1]}" if isinstance(weakest, (list, tuple)) and weakest else txt(weakest)
    lines.append("score:")
    lines.append(f"  S: {s_txt}")
    lines.append(f"  A: {_n(a, 4)}")
    lines.append(f"  s_E: {_n(score.get('s_E'), 4)}")
    lines.append(f"  eta_ratio: {_n(score.get('eta_ratio'), 4)}")
    lines.append(f"  weakest_class: {weakest_txt}")
    return "\n".join(lines)
