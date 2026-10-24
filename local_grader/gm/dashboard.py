"""Dashboard: state/run-detail JSON builders, HTTP server, static export.

Contract: INTERFACES2.md section 4. Owner: Agent S.
"""
import csv
import json
import math
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

from . import dashboard_charts, dashboard_page, history
from .config import CHECKED, TRACKS

PROGRESS_STALE_S = 60.0

# file name -> Content-Type; the only files served from a run directory
_FILE_TYPES = {
    "report.html": "text/html; charset=utf-8",
    "summary.json": "application/json; charset=utf-8",
    "power.csv": "text/csv; charset=utf-8",
}


# ── sanitising ────────────────────────────────────────────────────────────────
def _sanitize(obj):
    """Recursively replace float inf/nan with None (json.dumps allow_nan-safe)."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _parse_dt(value):
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


# ── state ─────────────────────────────────────────────────────────────────────
def _progress_block(store, row) -> dict:
    """Progress payload for one running local row (row = history ROW)."""
    run_id = str(row.get("id"))
    data = {}
    try:
        path = store.run_dir(run_id) / "progress.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    started = data.get("started_at") if isinstance(data.get("started_at"), str) else None
    updated = data.get("updated_at") if isinstance(data.get("updated_at"), str) else None
    d_start, d_upd = _parse_dt(started), _parse_dt(updated)
    now = datetime.now()
    elapsed = (now - d_start).total_seconds() if d_start is not None else None
    stale = d_upd is not None and (now - d_upd).total_seconds() > PROGRESS_STALE_S
    return {
        "run_id": run_id,
        "label": row.get("label"),
        "track": row.get("track"),
        "started_at": started,
        "elapsed_s": elapsed,
        "items_done": data.get("items_done"),
        "items_total": data.get("items_total"),
        "stale": bool(stale),
    }


def build_state(store, track) -> dict:
    """Assemble the /api/state payload for one track (JSON-serialisable)."""
    track = str(track)
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}; expected one of {sorted(TRACKS)}")
    rows = history.load_rows(store, track)
    baselines = {"local": None, "official": None}
    for r in rows:
        if r.get("is_baseline") and r["source"] in baselines and baselines[r["source"]] is None:
            baselines[r["source"]] = r
    running = next((r for r in rows
                    if r["source"] == "local" and r.get("status") == "running"), None)
    return _sanitize({
        "track": track,
        "generated_at": datetime.now().isoformat(),
        "progress": _progress_block(store, running) if running is not None else None,
        "baselines": baselines,
        "best": history.best_rows(rows, track),
        "history": rows,
        "groups": history.group_repeats(rows),
        "noise": history.noise_spread(rows),
        "calibration": history.calibration(rows, track),
        "charts": {
            "s_over_time": dashboard_charts.s_over_time(rows),
            "scatter": dashboard_charts.scatter_j_vs_slo(rows),
            "eta_bars": dashboard_charts.eta_bars(rows),
        },
    })


# ── run detail ────────────────────────────────────────────────────────────────
def _read_power(store, run_id) -> list:
    """runs/<id>/power.csv -> [[t_s, watts], ...] (header/junk rows skipped; [] if missing)."""
    path = store.run_dir(run_id) / "power.csv"
    if not path.is_file():
        return []
    out = []
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            for rec in csv.reader(fh):
                if len(rec) < 2:
                    continue
                try:
                    t, w = float(rec[0]), float(rec[1])
                except ValueError:
                    continue
                if math.isnan(t) or math.isinf(t) or math.isnan(w) or math.isinf(w):
                    continue
                out.append([t, w])
    except OSError:
        return []
    return out


def build_run_detail(store, run_id) -> dict:
    """Assemble the /api/run/<id> payload (JSON-serialisable)."""
    rid = store.resolve(run_id)
    run = store.get_run(rid)
    track = run.get("track")
    rows = history.load_rows(store, track) if track else history.load_rows(store)
    row = next((r for r in rows if r.get("key") == f"local:{rid}"), None)
    if row is None:
        row = history.local_row(run)
    summary = run.get("summary") if isinstance(run.get("summary"), dict) else None
    score = run.get("score") if isinstance(run.get("score"), dict) else None
    server = run.get("server") if isinstance(run.get("server"), dict) else None
    settings = run.get("settings") if isinstance(run.get("settings"), dict) else None

    class_rows = []
    fail_by_phase = {}
    tokens_by_class = {}
    if summary:
        class_acc = summary.get("class_acc") or {}
        class_n = summary.get("class_n") or {}
        base = store.get_baseline(track) if track else None
        base_summary = base.get("summary") if base and isinstance(base.get("summary"), dict) else None
        base_class_acc = (base_summary or {}).get("class_acc") or {}
        for cls, acc in class_acc.items():
            base_acc = base_class_acc.get(cls)
            ratio = acc / base_acc if (acc is not None and base_acc not in (None, 0)) else None
            checked = cls in CHECKED
            flag = "low" if (checked and ratio is not None and ratio < 0.75) else "ok"
            class_rows.append({"cls": cls, "acc": acc, "n": class_n.get(cls),
                               "base_acc": base_acc, "ratio": ratio,
                               "checked": checked, "flag": flag})
        by_phase = summary.get("by_phase") or {}
        for phase, info in by_phase.items():
            fail_by_phase[phase] = (info or {}).get("fails") or {}
        by_class = summary.get("by_class") or {}
        for cls, info in by_class.items():
            if isinstance(info, dict):
                tokens_by_class[cls] = {"mean": info.get("completion_tokens_mean"),
                                        "p95": info.get("completion_tokens_p95")}

    baseline_id = score.get("baseline_id") if score else None
    if baseline_id is None and run.get("is_baseline"):
        baseline_id = rid

    power = _read_power(store, rid)
    power_svg = dashboard_charts.power_chart(power, (summary or {}).get("bounds"))

    return _sanitize({
        "row": row,
        "summary": summary,
        "score": score,
        "server": server,
        "settings": settings,
        "baseline_id": baseline_id,
        "class_rows": class_rows,
        "fail_by_phase": fail_by_phase,
        "tokens_by_class": tokens_by_class,
        "power_svg": power_svg,
        "report_url": f"/runs/{rid}/report.html",
    })


# ── HTTP ──────────────────────────────────────────────────────────────────────
class _Handler(BaseHTTPRequestHandler):
    """GET-only dashboard handler. `store` is attached by make_server."""

    store = None
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # silence default request logging
        pass

    # -- helpers ---------------------------------------------------------------
    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    # -- routing -----------------------------------------------------------------
    def do_GET(self):
        try:
            self._route()
        except Exception as exc:  # keep the server alive on handler errors
            try:
                self._send_json(500, {"error": str(exc)})
            except Exception:
                pass

    def _route(self) -> None:
        store = type(self).store
        parts = urlsplit(self.path)
        raw = parts.path
        if raw == "/" or raw == "":
            self._send(200, dashboard_page.render_page().encode("utf-8"),
                       "text/html; charset=utf-8")
            return
        if raw == "/api/state":
            params = parse_qs(parts.query)
            track = (params.get("track") or ["val"])[0]
            if track not in TRACKS:
                self._send_json(400, {"error": f"unknown track {track!r}"})
                return
            self._send_json(200, build_state(store, track))
            return
        if raw == "/api/runs":
            rows = [r for r in history.load_rows(store) if r["source"] == "local"]
            self._send_json(200, _sanitize(rows))
            return
        if raw == "/api/official":
            rows = [r for r in history.load_rows(store) if r["source"] == "official"]
            self._send_json(200, _sanitize(rows))
            return
        if raw.startswith("/api/run/"):
            ref = unquote(raw[len("/api/run/"):])
            try:
                self._send_json(200, build_run_detail(store, ref))
            except ValueError as exc:
                self._send_json(404, {"error": str(exc)})
            return
        if raw.startswith("/runs/"):
            self._run_file(raw[len("/runs/"):])
            return
        self._send_json(404, {"error": f"not found: {raw}"})

    def _run_file(self, rest: str) -> None:
        store = type(self).store
        segs = rest.split("/")
        if len(segs) != 2 or not segs[0] or not segs[1]:
            self._send_json(404, {"error": "bad run path"})
            return
        rid, fname = segs
        for seg in (rid, fname):
            if "/" in seg or "\\" in seg or ".." in seg:
                self._send_json(404, {"error": "invalid path"})
                return
        if fname not in _FILE_TYPES:
            self._send_json(404, {"error": f"file {fname!r} not served"})
            return
        try:
            base = Path(store.runs_dir).resolve()
            target = (base / rid / fname).resolve()
            if base not in target.parents or not target.is_file():
                self._send_json(404, {"error": "not found"})
                return
        except (OSError, ValueError):
            self._send_json(404, {"error": "not found"})
            return
        self._send(200, target.read_bytes(), _FILE_TYPES[fname])


def make_server(store, host: str = "0.0.0.0", port: int = 8050) -> ThreadingHTTPServer:
    """ThreadingHTTPServer with the dashboard handler; port 0 → OS-assigned port."""
    class Handler(_Handler):
        pass
    Handler.store = store
    return ThreadingHTTPServer((host, port), Handler)


def serve(store, host: str = "0.0.0.0", port: int = 8050) -> None:
    """Run the dashboard server until Ctrl-C (clean exit)."""
    server = make_server(store, host, port)
    print(f"dashboard on http://{host}:{port if host != '0.0.0.0' else '127.0.0.1'}"
          f" (store {Path(store.root)})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# ── static export ─────────────────────────────────────────────────────────────
def export_static(store, path) -> Path:
    """One self-contained html snapshot (both track states + every run detail)."""
    states = {t: build_state(store, t) for t in ("val", "final")}
    runs = {}
    for run in store.list_runs(limit=10 ** 6):
        if run["status"] in ("done", "interrupted", "failed", "running"):
            runs[run["id"]] = build_run_detail(store, run["id"])
    html = dashboard_page.render_page(export={"states": states, "runs": runs})
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    return p
