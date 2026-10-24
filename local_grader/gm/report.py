"""Self-contained HTML run report + two-run text diff (Agent B).

`build_report(store, run_id)` renders ONE self-contained .html file (inline <style>, inline SVG, no external
resources, no JS) into runs/<id>/report.html. `diff_text(store, id_a, id_b)` renders aligned key-metric deltas.

The `store` object is injected (never imported here) and only the INTERFACES.md surface is used:
get_run, run_dir, read_json, read_jsonl, get_baseline, resolve.
"""
import csv
import html as _html
import math
from pathlib import Path

from . import config, score as score_mod, slo

_SVG_W, _SVG_H = 800.0, 320.0
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 48.0, 14.0, 14.0, 34.0

_STYLE = """
:root { --bg: #ffffff; --text: #1c1e21; --muted: #5c6670; --grid: #d8dce1; --axis: #8a94a0;
        --accent: #1457a0; --accent-soft: rgba(20, 87, 160, 0.12); --band: rgba(20, 87, 160, 0.10);
        --hl: #b3261e; --ok: #1a7f37; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0e1218; --text: #e6e9ee; --muted: #9aa5b1; --grid: #2b3440; --axis: #4a5562;
         --accent: #6ea8ff; --accent-soft: rgba(110, 168, 255, 0.14); --band: rgba(110, 168, 255, 0.12);
         --hl: #ff8a80; --ok: #3fb950; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 24px 20px 60px; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 28px 0 10px; }
h3 { font-size: 1rem; margin: 18px 0 8px; }
p, li, td, th { line-height: 1.45; }
table.grid { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 0.9rem; }
table.grid th, table.grid td { border: 1px solid var(--grid); padding: 5px 8px; text-align: left; vertical-align: top; }
table.grid th { background: var(--accent-soft); }
table.grid th.hl, table.grid td.hl { background: var(--hl); color: #ffffff; font-weight: 700; }
td.mono, th.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
td.num, th.num { text-align: right; }
.fact { display: grid; grid-template-columns: 170px 1fr; gap: 4px 12px; margin: 0 0 8px; }
.fact dt { color: var(--muted); font-weight: 600; }
.fact dd { margin: 0; overflow-wrap: anywhere; }
table.kv { border-collapse: collapse; margin: 6px 0 10px; font-size: 0.9rem; }
table.kv th, table.kv td { border: 1px solid var(--grid); padding: 4px 8px; }
.score-card { border: 1px solid var(--grid); border-radius: 10px; padding: 14px 18px; margin: 10px 0 6px;
             background: var(--accent-soft); }
.score-value { font-size: 1.6rem; font-weight: 700; }
.score-facts { display: flex; flex-wrap: wrap; gap: 8px 28px; margin-top: 8px; }
.score-facts div { display: inline-block; }
.score-facts dt { color: var(--muted); font-weight: 600; }
.score-facts dd { margin: 0; font-size: 1.15rem; font-weight: 600; }
.note { color: var(--muted); font-style: italic; }
.baseline-note { margin: 8px 0 0; }
.hl { background: var(--hl); color: #ffffff; font-weight: 700; }
svg.power-chart { width: 100%; height: auto; display: block; background: var(--bg); }
svg.power-chart .grid { stroke: var(--grid); stroke-width: 1; }
svg.power-chart .axis { stroke: var(--axis); stroke-width: 1.5; }
svg.power-chart .tick { fill: var(--muted); font-size: 12px; }
svg.power-chart .power-line { stroke: var(--accent); stroke-width: 2; }
svg.power-chart .arrival { stroke: var(--hl); stroke-width: 2; }
svg.power-chart .band { stroke: none; }
footer { margin-top: 34px; color: var(--muted); font-size: 0.8rem; }
"""


def _esc(value):
    return _html.escape("" if value is None else str(value), quote=True)


def _f(x, digits=4):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) else x


def _fmt(x, digits=3):
    x = _f(x, digits)
    if x is None:
        return "n/a"
    if math.isinf(x):
        return "inf"
    s = f"{x:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def _dl(a, b):
    """delta = B - A; None when either side is missing."""
    if a is None or b is None:
        return None
    return float(b) - float(a)


def _delta_str(d, digits=4):
    if d is None:
        return "-"
    d = float(d)
    if d.is_integer():
        return f"{d:+.0f}"
    return f"{d:+.{digits}f}"


def _fails_text(fails):
    if not fails:
        return "—"
    if isinstance(fails, dict):
        return ", ".join(f"{k}: {v}" for k, v in fails.items())
    return ", ".join(str(f) for f in fails)


def _store_read_json(store, run_id, name):
    """Tolerate both bare stems and filenames for the injected store (read_json(run_id, name))."""
    for n in (name, f"{name}.json"):
        try:
            return store.read_json(run_id, n)
        except Exception:
            continue
    return None


def _store_read_jsonl(store, run_id, name):
    """Tolerate both bare stems and filenames for the injected store (read_jsonl(run_id, name))."""
    for n in (name, f"{name}.jsonl"):
        try:
            return store.read_jsonl(run_id, n)
        except Exception:
            continue
    return []


def _html_table(headers, rows):
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for row in rows:
        cells = "".join(f"<td>{_esc(c)}</td>" for c in row)
        body.append(f"<tr>{cells}</tr>")
    return f'<table class="grid"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def _phase_rows(summary):
    by_phase = summary.get("by_phase") or {}
    names = [p for p in config.PHASES if p in by_phase] + sorted(set(by_phase) - set(config.PHASES))
    rows = []
    for p in names:
        g = by_phase.get(p) or {}
        rows.append([p, str(g.get("n")), str(g.get("on_time")), _fails_text(g.get("fails")),
                     _fmt(g.get("p95_tpot_ms")), _fmt(g.get("p95_ttft_ratio")), str(g.get("cutoffs"))])
    return rows


def _slo_class_rows(summary):
    by_class = summary.get("by_class") or {}
    rows = []
    for c in sorted(set(list(by_class) + list(config.CHECKED))):
        g = by_class.get(c) or {}
        rows.append([c, str(g.get("n")), str(g.get("on_time")), _fails_text(g.get("fails")),
                     _fmt(g.get("p95_tpot_ms")), _fmt(g.get("p95_ttft_ratio")), str(g.get("cutoffs"))])
    return rows


def _token_rows(summary):
    by_class = summary.get("by_class") or {}
    rows = []
    for c in sorted(set(list(by_class) + list(config.CHECKED))):
        g = by_class.get(c) or {}
        rows.append([c, str(g.get("n")), _fmt(g.get("completion_tokens_mean")),
                     _fmt(g.get("completion_tokens_p95")), str(g.get("cutoffs"))])
    return rows


# ───────────────────────────── power chart ─────────────────────────────

def _read_power(store, run_id):
    """→ [(t_s, watts, joules_cum)] from runs/<id>/power.csv via the csv module."""
    path = store.run_dir(run_id) / "power.csv"
    pts = []
    if not path.exists():
        return pts
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                pts.append((float(row["t_s"]), float(row["watts"]), float(row["joules_cum"])))
            except (KeyError, TypeError, ValueError):
                continue
    return pts


def _downsample(points, limit=600):
    if len(points) <= limit:
        return list(points)
    step = math.ceil(len(points) / limit)
    return [points[i] for i in range(0, len(points), step)]


def _power_svg(points, bounds, arrivals, end_t):
    """Inline SVG: watts vs t_s line, shaded phase bands, arrival ticks, labelled axes."""
    W, H = _SVG_W, _SVG_H
    pad_l, pad_r, pad_t, pad_b = _PAD_L, _PAD_R, _PAD_T, _PAD_B
    iw, ih = W - pad_l - pad_r, H - pad_t - pad_b
    ts = [p[0] for p in points] or [0.0]
    ws = [p[1] for p in points] or [0.0]
    t0, t1 = min(ts), max(ts)
    if t1 <= t0:
        t1 = t0 + 1.0
    wmax = max(max(ws), 0.0)
    yhi = wmax * 1.2 if wmax > 0 else 10.0
    x = lambda t: pad_l + (t - t0) / (t1 - t0) * iw
    y = lambda w: pad_t + (1.0 - w / yhi) * ih

    d = f"M {x(points[0][0]):.1f} {y(points[0][1]):.1f}"
    for p in points[1:]:
        d += f" L {x(p[0]):.1f} {y(p[1]):.1f}"

    nx, ny = 5, 4
    xticks = [t0 + (t1 - t0) * i / nx for i in range(nx + 1)]
    yticks = [yhi * i / ny for i in range(ny + 1)]

    bands = []
    quiet_end = 0.0
    bmap = bounds if isinstance(bounds, dict) else {}
    quiet = bmap.get("quiet") or (0.0, 0.0)
    if isinstance(quiet, (list, tuple)) and len(quiet) >= 2:
        quiet_end = float(quiet[1])
    window = float(end_t) if end_t else t1
    for p in ("ramp", "peak", "quiet", "drain"):
        if p == "drain":
            a, b = quiet_end, window
        else:
            ab = bmap.get(p) or (0.0, 0.0)
            a, b = float(ab[0]), float(ab[1])
        if b > a:
            bands.append((p, a, min(b, window)))

    out = []
    out.append(f'<svg class="power-chart" viewBox="0 0 {int(_SVG_W)} {int(_SVG_H)}" width="{_SVG_W:.0f}" '
               f'height="{_SVG_H:.0f}" role="img" aria-label="power over time">')
    for p, a, b in bands:
        if b <= a:
            continue
        xa, xb = x(a), x(b)
        out.append(f'<rect class="band band-{p}" x="{xa:.1f}" y="{pad_t:.1f}" '
                   f'width="{max(0.0, xb - xa):.1f}" height="{ih:.1f}"/>')
    for w in yticks:
        yy = y(w)
        out.append(f'<line class="grid" x1="{pad_l}" x2="{W - pad_r}" y1="{yy:.1f}" y2="{yy:.1f}"/>')
        out.append(f'<text class="tick" x="{pad_l - 6:.1f}" y="{yy + 4:.1f}" text-anchor="end">{_esc(_fmt(w, 0))} W</text>')
    for t in xticks:
        xx = x(t)
        out.append(f'<line class="grid" x1="{xx}" x2="{xx}" y1="{pad_t}" y2="{H - pad_b}"/>')
        out.append(f'<text class="tick" x="{xx}" y="{H - pad_b + 16:.1f}" text-anchor="middle">{_esc(_fmt(t, 0))} s</text>')
    out.append(f'<line class="axis" x1="{pad_l}" x2="{W - pad_r}" y1="{H - pad_b}" y2="{H - pad_b}"/>')
    out.append(f'<path class="power-line" d="{_esc(d)}" fill="none"/>')
    for t in sorted({float(a) for a in arrivals if a is not None}):
        if t < t0 or t > t1:
            continue
        xx = x(t)
        out.append(f'<line class="arrival" x1="{xx:.1f}" x2="{xx:.1f}" y1="{H - pad_b}" y2="{H - pad_b - 8:.1f}"/>')
    out.append("</svg>")
    return "\n".join(out)


# ───────────────────────────── HTML rendering ─────────────────────────────

def _render(run, summary, score, settings, records, baseline_run, store, run_id):
    server = run.get("server") or {}
    is_baseline = bool(run.get("is_baseline")) or bool(score.get("is_baseline"))
    sc = _f(score.get("S"))
    s_disp = "0.00" if is_baseline else (_fmt(score.get("S")) if sc is not None else "n/a")
    baseline_id = score.get("baseline_id") or (baseline_run or {}).get("id") or run.get("id")

    parts = []
    parts.append("<!DOCTYPE html>")
    parts.append('<html lang="en">')
    parts.append("<head>")
    parts.append('<meta charset="utf-8">')
    parts.append("<title>Run report</title>")
    parts.append("<style>")
    parts.append(_STYLE)
    parts.append("</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append("<main>")
    parts.append("<h1>Run report</h1>")
    parts.append(f"<p class=\"muted\">Run <strong>{_esc(run.get('id'))}</strong> — self-contained local report.</p>")

    # 1. header
    parts.append("<h2>1. Header</h2>")
    parts.append('<dl class="fact">')
    for k, v in [
        ("Label", run.get("label")),
        ("Track", run.get("track")),
        ("Status", run.get("status")),
        ("Started", run.get("started_at")),
        ("Ended", run.get("ended_at")),
        ("Model", server.get("model") or server.get("model_name") or server.get("id")),
        ("Server command", server.get("vllm_cmdline")),
        ("Power limit / clocks", server.get("nvidia_smi")),
    ]:
        parts.append(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>")
    parts.append("</dl>")
    parts.append("<h3>Calibration settings</h3>")
    parts.append('<table class="kv"><thead><tr><th>key</th><th>value</th></tr></thead><tbody>')
    for k in sorted(settings or {}):
        parts.append(f"<tr><td>{_esc(k)}</td><td>{_esc(settings.get(k))}</td></tr>")
    parts.append("</tbody></table>")

    # 2. score card
    parts.append("<h2>2. Score card</h2>")
    parts.append('<div class="score-card">')
    parts.append(f"<div class=\"score-value\">S = <span>{_esc(s_disp)}</span></div>")
    parts.append('<div class="score-facts">')
    parts.append(f"<div><dt>A</dt><dd>{_esc(_fmt((score or {}).get('A')))}</dd></div>")
    parts.append(f"<div><dt>s_E</dt><dd>{_esc(_fmt((score or {}).get('s_E')))}</dd></div>")
    parts.append(f"<div><dt>eta ratio</dt><dd>{_esc(_fmt((score or {}).get('eta_ratio')))}</dd></div>")
    parts.append(f"<div><dt>s_T</dt><dd>{_esc(_fmt((summary or {}).get('s_T')))}</dd></div>")
    parts.append(f"<div><dt>s_L</dt><dd>{_esc(_fmt((summary or {}).get('s_L')))}</dd></div>")
    parts.append("</div></div>")
    parts.append(f'<p class="baseline-note">Baseline run: <strong>{_esc(baseline_id)}</strong>'
                 f"{' — baseline (S = 0 by definition)' if is_baseline else ''}.</p>")

    # 3. SLO
    parts.append("<h2>3. SLO</h2>")
    overall = {
        "n": (summary or {}).get("requests"),
        "on_time": (summary or {}).get("on_time"),
        "fails": (summary or {}).get("fail_reasons") or {},
        "p50_ttft_ratio": (summary or {}).get("p50_ttft_over_slo"),
        "p95_ttft_ratio": (summary or {}).get("p95_ttft_over_slo"),
        "p50_tpot_ms": (summary or {}).get("p50_tpot_ms"),
        "p95_tpot_ms": (summary or {}).get("p95_tpot_ms"),
        "cutoffs": (summary or {}).get("cutoffs"),
    }
    slo_rows = [[f"{overall['on_time']}/{overall['n']}", _fails_text(overall["fails"]),
                 _fmt(overall["p50_ttft_ratio"]), _fmt(overall["p95_ttft_ratio"]),
                 _fmt(overall["p50_tpot_ms"]), _fmt(overall["p95_tpot_ms"]), _fmt(overall["cutoffs"])]]
    parts.append("<h3>Overall</h3>")
    parts.append(_html_table(["on-time / total", "fail reasons", "p50 ttft/slo", "p95 ttft/slo",
                              "p50 tpot ms", "p95 tpot ms", "cutoffs"], slo_rows))
    parts.append("<h3>By phase</h3>")
    parts.append(_html_table(["phase", "n", "on-time", "fails", "p95 tpot ms", "p95 ttft/slo", "cutoffs"],
                             _phase_rows(summary or {})))
    parts.append("<h3>By class</h3>")
    parts.append(_html_table(["class", "n", "on-time", "fails", "p95 tpot ms", "p95 ttft/slo", "cutoffs"],
                             _slo_class_rows(summary or {})))

    # 4. energy
    parts.append("<h2>4. Energy</h2>")
    energy_J = (summary or {}).get("energy_J")
    parts.append('<table class="kv"><thead><tr><th>metric</th><th>value</th></tr></thead><tbody>')
    parts.append(f"<tr><th>total energy (J)</th><td>{_esc(_fmt(energy_J))}</td></tr>")
    parts.append(f"<tr><th>avg power (W)</th><td>{_esc(_fmt((summary or {}).get('avg_power_W')))}</td></tr>")
    parts.append(f"<tr><th>idle power (W)</th><td>{_esc(_fmt((summary or {}).get('idle_W')))}</td></tr>")
    parts.append(f"<tr><th>tokens / J</th><td>{_esc(_fmt((summary or {}).get('tokens_per_J')))}</td></tr>")
    phase_energy = (summary or {}).get("phase_energy_J") or {}
    parts.append("<tr><th>joules per phase</th><td>"
                 + ", ".join(f"{k}: {_fmt(v)}" for k, v in phase_energy.items()) + "</td></tr>")
    parts.append("</tbody></table>")
    if energy_J is None or (summary or {}).get("energy_ok") is False:
        parts.append('<p class="note">Energy data unavailable (NVML missing): metrics above are n/a and the '
                     "power chart is omitted.</p>")
    else:
        points = _downsample(_read_power(store, run_id))
        arrivals = [r.get("sched_t") for r in (records or []) if (r.get("turn") or 0) == 0]
        svg = _power_svg(points, (summary or {}).get("bounds"), arrivals, (summary or {}).get("window_s")) if points else ""
        if svg:
            parts.append(svg)
        else:
            parts.append('<p class="note">No power samples available.</p>')

    # 5. accuracy
    parts.append("<h2>5. Accuracy</h2>")
    acc = (summary or {}).get("class_acc") or {}
    class_n = (summary or {}).get("class_n") or {}
    class_ratio = (score or {}).get("class_ratio") or {}
    baseline_acc = None
    if baseline_run is not None:
        baseline_acc = (baseline_run.get("summary") or {}).get("class_acc") or {}
    acc_rows = []
    for c in sorted(set(list(acc) + list(baseline_acc or {}))):
        a = acc.get(c)
        base = baseline_acc.get(c) if baseline_acc else None
        ratio = class_ratio.get(c) if c in class_ratio else (None if not base else (a / base if base else None))
        checked = c in config.CHECKED
        hot = checked and ratio is not None and ratio < 0.75
        acc_rows.append([c, a, class_n.get(c), base, ratio, hot])
    parts.append('<table class="grid"><thead><tr>'
                 '<th>class</th><th>accuracy</th><th>n</th><th>baseline acc</th>'
                 '<th>ratio vs baseline</th><th>flag</th></tr></thead><tbody>')
    for r in acc_rows:
        flag = f'<th class="hl">{_esc("CHECKED < 0.75")}</th>' if r[5] else "<th></th>"
        parts.append("<tr>")
        parts.append(f"<td>{_esc(r[0])}</td>")
        parts.append(f"<td>{_esc(_fmt(r[1]))}</td>")
        parts.append(f"<td>{_esc(str(r[2]))}</td>")
        parts.append(f"<td>{_esc(_fmt(r[3]))}</td>")
        parts.append(f"<td>{_esc(_fmt(r[4]))}</td>")
        parts.append(flag)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    parts.append(f"<p>Macro accuracy: {_esc(_fmt((summary or {}).get('macro_acc')))}</p>")

    # 6. completion tokens
    parts.append("<h2>6. Completion tokens</h2>")
    parts.append(_html_table(["class", "n", "mean completion tokens", "p95 completion tokens", "cutoffs"],
                              _token_rows(summary or {})))

    # 7. worst 20
    parts.append("<h2>7. Worst 20 requests</h2>")
    rows = slo.effective(records or [], settings or {})
    ordered = sorted(rows, key=lambda r: (0 if math.isinf(r.get("ttft_ratio") or math.inf) else 1,
                                         -(r.get("ttft_ratio") or math.inf),
                                         0 if math.isinf(r.get("tpot") or math.inf) else 1,
                                         -(r.get("tpot") or math.inf)))
    worst_rows = []
    for r in ordered[:20]:
        rid, turn = r.get("id"), r.get("turn") or 0
        rs = [x for x in (records or []) if str(x.get("id")) == str(rid) and (x.get("turn") or 0) == turn]
        if not rs:
            rs = [x for x in (records or []) if str(x.get("id")) == str(rid) and (x.get("turn") or 0) == 0]
        tail = " ".join(str(x.get("answer_tail") or "") for x in rs)
        tpot_ms = (r.get("tpot") * 1000) if r.get("tpot") is not None else None
        worst_rows.append([rid, r.get("cls"), str(r.get("turn") or 0), r.get("phase"),
                           _fmt(r.get("ttft")), _fmt(r.get("ttft_ratio")), _fmt(tpot_ms),
                           _fails_text(r.get("fails")), tail])
    parts.append(_html_table(["id", "class", "turn", "phase", "ttft", "ttft ratio", "tpot ms", "fails",
                              "answer tail"], worst_rows))

    parts.append("</main>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


# ───────────────────────────── public API ─────────────────────────────

def _effective_settings(overrides):
    try:
        return config.resolve_settings(overrides)
    except Exception:
        return dict(config.SETTING_DEFAULTS)


def build_report(store, run_id) -> Path:
    """Write runs/<id>/report.html; return its path."""
    run = store.get_run(run_id) or {}
    summary = run.get("summary") or {}
    score = run.get("score") or {}
    settings = _effective_settings(run.get("settings"))
    records = _store_read_jsonl(store, run_id, "records") or []
    baseline_run = store.get_baseline(run.get("track")) if run.get("track") else None
    html_text = _render(run, summary, score, settings, records, baseline_run, store, run_id)
    path = store.run_dir(run_id) / "report.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path


def diff_text(store, id_a, id_b) -> str:
    a = store.get_run(id_a) or {}
    b = store.get_run(id_b) or {}
    sa, sb = a.get("summary") or {}, b.get("summary") or {}
    sca, scb = a.get("score") or {}, b.get("score") or {}

    rows = [
        ("on-time", f"{sa.get('on_time')}/{sa.get('requests')}", f"{sb.get('on_time')}/{sb.get('requests')}",
         _dl(sa.get("on_time_frac"), sb.get("on_time_frac"))),
        ("requests", str(sa.get("requests")), str(sb.get("requests")), _dl(sa.get("requests"), sb.get("requests"))),
        ("S", _fmt(sca.get("S")), _fmt(scb.get("S")), _dl(sca.get("S"), scb.get("S"))),
        ("A", _fmt(sca.get("A")), _fmt(scb.get("A")), _dl(sca.get("A"), scb.get("A"))),
        ("s_E", _fmt(sca.get("s_E")), _fmt(scb.get("s_E")), _dl(sca.get("s_E"), scb.get("s_E"))),
        ("eta", _fmt(sa.get("eta")), _fmt(sb.get("eta")), _dl(sa.get("eta"), sb.get("eta"))),
        ("s_T", _fmt(sa.get("s_T")), _fmt(sb.get("s_T")), _dl(sa.get("s_T"), sb.get("s_T"))),
        ("s_L", _fmt(sa.get("s_L")), _fmt(sb.get("s_L")), _dl(sa.get("s_L"), sb.get("s_L"))),
        ("p95 ttft ratio", _fmt(sa.get("p95_ttft_over_slo")), _fmt(sb.get("p95_ttft_over_slo")),
         _dl(sa.get("p95_ttft_over_slo"), sb.get("p95_ttft_over_slo"))),
        ("p95 tpot ms", _fmt(sa.get("p95_tpot_ms")), _fmt(sb.get("p95_tpot_ms")),
         _dl(sa.get("p95_tpot_ms"), sb.get("p95_tpot_ms"))),
        ("energy J", _fmt(sa.get("energy_J")), _fmt(sb.get("energy_J")), _dl(sa.get("energy_J"), sb.get("energy_J"))),
        ("avg W", _fmt(sa.get("avg_power_W")), _fmt(sb.get("avg_power_W")), _dl(sa.get("avg_power_W"), sb.get("avg_power_W"))),
        ("tokens/J", _fmt(sa.get("tokens_per_J")), _fmt(sb.get("tokens_per_J")), _dl(sa.get("tokens_per_J"), sb.get("tokens_per_J"))),
        ("macro accuracy", _fmt(sa.get("macro_acc")), _fmt(sb.get("macro_acc")), _dl(sa.get("macro_acc"), sb.get("macro_acc"))),
    ]

    lines = [f"Report diff: {id_a}  vs  {id_b}", ""]
    lines.append(f"{'metric':<20}{'A':>14}{'B':>14}{'delta':>12}")
    lines.append("-" * 52)
    for m, va, vb, d in rows:
        lines.append(f"{m:<20}{str(va):>14}{str(vb):>14}{_delta_str(d):>12}")

    acc_a = sa.get("class_acc") or {}
    acc_b = sb.get("class_acc") or {}
    lines.append("")
    lines.append("per-class accuracy (A | B | delta):")
    lines.append(f"{'class':<20}{'A acc':>12}{'B acc':>12}{'delta':>12}")
    lines.append("-" * 40)
    for c in sorted(set(list(acc_a) + list(acc_b))):
        a, b = acc_a.get(c), acc_b.get(c)
        lines.append(f"{c:<20}{_fmt(a) if a is not None else '-':>12}{_fmt(b) if b is not None else '-':>12}"
                     f"{_delta_str(_dl(a, b)):>12}")

    lines.append("")
    lines.append("per-class on-time (A | B | delta):")
    lines.append(f"{'class':<20}{'A on-time':>12}{'B on-time':>12}{'delta':>12}")
    lines.append("-" * 40)
    by_a = sa.get("by_class") or {}
    by_b = sb.get("by_class") or {}
    for c in sorted(set(list(by_a) + list(by_b))):
        da = (by_a.get(c) or {}).get("on_time")
        db = (by_b.get(c) or {}).get("on_time")
        lines.append(f"{c:<20}{str(da) if da is not None else '-':>12}"
                     f"{str(db) if db is not None else '-':>12}"
                     f"{_delta_str(_dl(da, db)):>12}")
    lines.append("")
    return "\n".join(lines)
