"""Dashboard page (Agent F).

``PAGE_HTML`` is the complete live dashboard document.  ``render_page(export=None)`` returns that
document unchanged for live use, or a static snapshot with the state/run JSON embedded in a
``<script id="export-data" type="application/json">`` block.  All data access in the browser goes
through ``api(path, callback)`` which speaks XMLHttpRequest in live mode and reads the embedded
snapshot in export mode.

The dashboard's HTTP API (``/api/state`` and ``/api/run/<id>``) is documented in
INTERFACES2.md section 4; this module only renders the page and never imports the server.
"""
import json

__all__ = ["PAGE_HTML", "render_page"]


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GreenMind local grader</title>
<style>
:root {
  --fg: #18212f;
  --bg: #f6f8fb;
  --muted: #64748b;
  --grid: #d8dee8;
  --local: #2563eb;
  --official: #d97706;
  --accent: #059669;
  --green: #16a34a;
  --amber: #d97706;
  --red: #dc2626;
  --card: #ffffff;
  --soft: rgba(37, 99, 235, 0.08);
  --official-soft: rgba(217, 119, 6, 0.08);
  --red-soft: rgba(220, 38, 38, 0.10);
  --shadow: 0 1px 2px rgba(16, 24, 40, 0.07);
}
@media (prefers-color-scheme: dark) {
  :root {
    --fg: #e5e7eb;
    --bg: #0f172a;
    --muted: #94a3b8;
    --grid: #334155;
    --local: #60a5fa;
    --official: #fbbf24;
    --accent: #34d399;
    --green: #4ade80;
    --amber: #fbbf24;
    --red: #f87171;
    --card: #1e293b;
    --soft: rgba(96, 165, 250, 0.14);
    --official-soft: rgba(251, 191, 36, 0.12);
    --red-soft: rgba(248, 113, 113, 0.14);
    --shadow: none;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--fg);
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  font-size: 14px;
  line-height: 1.45;
}
header.top {
  background: var(--card);
  border-bottom: 1px solid var(--grid);
}
header.top .inner {
  max-width: 1200px;
  margin: 0 auto;
  padding: 12px 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
}
h1 { font-size: 1.25rem; margin: 0; letter-spacing: -0.01em; }
h2 { font-size: 1.05rem; margin: 26px 0 10px; }
h3 { font-size: 0.92rem; margin: 18px 0 6px; }
main { max-width: 1200px; margin: 0 auto; padding: 6px 20px 70px; }
.muted { color: var(--muted); }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
.num { text-align: right; }
.error { color: var(--red); font-weight: 600; }
#updated { margin-left: auto; font-size: 0.8rem; color: var(--muted); }
.tracks { display: inline-flex; gap: 6px; }
.track-btn {
  font: inherit;
  font-size: 0.85rem;
  padding: 5px 16px;
  border-radius: 999px;
  border: 1px solid var(--grid);
  background: transparent;
  color: var(--fg);
  cursor: pointer;
}
.track-btn.active { background: var(--accent); border-color: var(--accent); color: #ffffff; font-weight: 600; }
.status-strip { display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 4px; align-items: stretch; }
.progress-block {
  flex: 1 1 360px;
  background: var(--card);
  border: 1px solid var(--grid);
  border-radius: 10px;
  padding: 10px 14px;
  box-shadow: var(--shadow);
}
.progress-line { margin-bottom: 6px; }
.progress-line progress { width: 100%; height: 10px; }
.stale { color: var(--amber); font-weight: 600; }
.baseline-box {
  flex: 0 1 230px;
  background: var(--card);
  border: 1px solid var(--grid);
  border-radius: 10px;
  padding: 8px 12px;
  box-shadow: var(--shadow);
}
.box-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); }
.baseline-label { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.metric-row { display: flex; justify-content: space-between; gap: 10px; font-size: 0.82rem; }
.metric-row span:first-child { color: var(--muted); }
#best { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 12px; margin-top: 12px; }
.card {
  background: var(--card);
  border: 1px solid var(--grid);
  border-radius: 12px;
  padding: 12px 16px;
  box-shadow: var(--shadow);
}
.best-card { border-left: 4px solid var(--local); }
.best-card.official { border-left-color: var(--official); }
.card-title { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.best-value { font-size: 1.9rem; font-weight: 700; line-height: 1.15; }
.best-label { font-weight: 600; }
.best-server { color: var(--muted); font-size: 0.8rem; overflow-wrap: anywhere; }
.table-wrap {
  overflow-x: auto;
  background: var(--card);
  border: 1px solid var(--grid);
  border-radius: 10px;
  box-shadow: var(--shadow);
}
#history { border-collapse: collapse; width: 100%; font-size: 0.78rem; }
#history th, #history td { padding: 4px 7px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
#history thead th {
  position: sticky;
  top: 0;
  background: var(--card);
  color: var(--muted);
  font-weight: 600;
  text-align: left;
  user-select: none;
}
#history thead th[data-sort] { cursor: pointer; }
#history thead th[data-sort]:hover { color: var(--fg); }
#history th.num { text-align: right; }
#history thead th[data-dir="asc"]::after { content: " ▲"; color: var(--accent); }
#history thead th[data-dir="desc"]::after { content: " ▼"; color: var(--accent); }
#history tbody tr { cursor: pointer; }
#history tbody tr:hover { background: var(--soft); }
#history tbody tr.row-local { background: var(--soft); }
#history tbody tr.row-official { background: var(--official-soft); }
#history tbody tr.row-local td:first-child { border-left: 3px solid var(--local); }
#history tbody tr.row-official td:first-child { border-left: 3px solid var(--official); }
#history tbody tr.baseline { font-weight: 700; }
#history tbody tr.selected { outline: 2px solid var(--accent); outline-offset: -2px; }
#history td.config, #history td.pair { max-width: 260px; overflow: hidden; text-overflow: ellipsis; }
.badge {
  display: inline-block;
  padding: 0 6px;
  border-radius: 999px;
  font-size: 0.68rem;
  font-weight: 600;
  vertical-align: 1px;
}
.badge-base { background: var(--soft); color: var(--local); border: 1px solid var(--local); }
.badge-noise { background: var(--grid); color: var(--muted); }
.status { font-size: 0.72rem; }
.groups { margin-top: 8px; font-size: 0.82rem; }
.groups-title { font-weight: 600; color: var(--muted); margin-bottom: 2px; }
.group-line { margin: 1px 0; }
.noise { margin: 4px 0 0; font-size: 0.8rem; }
.cal-pair { margin: 10px 0 16px; }
.cal-pair h3 { margin: 0 0 4px; }
table.grid { border-collapse: collapse; width: 100%; font-size: 0.82rem; background: var(--card); }
table.grid th, table.grid td { border: 1px solid var(--grid); padding: 4px 8px; text-align: left; }
table.grid th { background: var(--soft); font-weight: 600; }
table.grid th.num, table.grid td.num { text-align: right; }
.cal-table { max-width: 640px; }
.rank-line { font-weight: 600; }
.pill {
  display: inline-block;
  min-width: 44px;
  text-align: center;
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 0.72rem;
  font-weight: 700;
  color: #ffffff;
}
.pill-green { background: var(--green); }
.pill-amber { background: var(--amber); }
.pill-red { background: var(--red); }
.pill-na { background: var(--grid); color: var(--muted); }
#charts { display: grid; gap: 12px; }
#charts figure {
  margin: 0;
  background: var(--card);
  border: 1px solid var(--grid);
  border-radius: 10px;
  padding: 10px 12px;
  box-shadow: var(--shadow);
}
#charts figcaption { color: var(--muted); font-size: 0.78rem; font-weight: 600; margin-bottom: 4px; }
#charts svg, .chart-box svg { max-width: 100%; height: auto; display: block; }
#detail {
  background: var(--card);
  border: 1px solid var(--grid);
  border-radius: 10px;
  padding: 4px 16px 16px;
  margin-top: 26px;
  box-shadow: var(--shadow);
}
.detail-links { margin: 4px 0 8px; }
.report-link { color: var(--accent); font-weight: 600; }
.server-summary { color: var(--muted); font-size: 0.8rem; overflow-wrap: anywhere; margin: 4px 0 10px; }
.facts { display: flex; flex-wrap: wrap; gap: 6px 26px; margin: 8px 0 12px; }
.fact { display: flex; gap: 8px; font-size: 0.85rem; }
.fact-k { color: var(--muted); }
.fact-v { font-weight: 600; }
.detail-table { margin: 4px 0 12px; }
.row-low td { background: var(--red-soft); }
.low-note { color: var(--red); font-weight: 700; }
.notes { white-space: pre-wrap; }
.chart-box { min-height: 20px; }
footer { color: var(--muted); font-size: 0.75rem; margin-top: 26px; }
</style>
</head>
<body>
<header class="top">
  <div class="inner">
    <h1>GreenMind local grader</h1>
    <div class="tracks" role="group" aria-label="Track">
      <button type="button" id="track-val" class="track-btn active">val</button>
      <button type="button" id="track-final" class="track-btn">final</button>
    </div>
    <span id="updated" class="muted">updated --:--:--</span>
  </div>
</header>
<main>
  <section id="status" class="status-strip"><span class="muted">Loading dashboard…</span></section>
  <section id="best"></section>

  <h2>History</h2>
  <div class="table-wrap">
    <table id="history">
      <thead>
        <tr>
          <th data-sort="source">Src</th>
          <th data-sort="label">Label</th>
          <th data-sort="started">Started</th>
          <th data-sort="S" class="num">S</th>
          <th data-sort="A" class="num">A</th>
          <th data-sort="R" class="num">R</th>
          <th data-sort="s_E" class="num">sE</th>
          <th data-sort="s_T" class="num">sT</th>
          <th data-sort="s_L" class="num">sL</th>
          <th data-sort="eta_vs_base" class="num">η/base</th>
          <th data-sort="j_per_req" class="num">J/req</th>
          <th data-sort="p95_ttft_s" class="num">p95 TTFT (s)</th>
          <th data-sort="p95_tpot_ms" class="num">p95 TPOT (ms)</th>
          <th data-sort="slo_frac" class="num">SLO</th>
          <th data-sort="accuracy_pct" class="num">Acc %</th>
          <th data-sort="is_baseline">Base</th>
          <th data-sort="pair_key">Pair</th>
          <th data-sort="server_summary">Config</th>
        </tr>
      </thead>
      <tbody id="history-body"></tbody>
    </table>
  </div>
  <div id="groups" class="groups"></div>
  <div id="noise" class="noise muted"></div>

  <section id="calibration" class="panel"></section>
  <section id="charts" class="panel">
    <figure>
      <figcaption>S over time</figcaption>
      <div id="chart-s-over-time" class="chart-box"><span class="muted">no data</span></div>
    </figure>
    <figure>
      <figcaption>J/request vs SLO met</figcaption>
      <div id="chart-scatter" class="chart-box"><span class="muted">no data</span></div>
    </figure>
    <figure>
      <figcaption>η vs baseline</figcaption>
      <div id="chart-eta-bars" class="chart-box"><span class="muted">no data</span></div>
    </figure>
  </section>

  <section id="detail"></section>
  <footer>Auto-refresh every 15 s (live mode). Values with no NVML energy data show as n/a.</footer>
</main>
@@EXPORT_DATA@@<script>
(function () {
  var EXPORT = @@EXPORT_FLAG@@;
  var TRACK = 'val';
  var stateData = null;
  var selectedKey = null;
  var detailCache = {};
  var sortKey = null;
  var sortDir = 1;
  var exportData = null;
  var TEXT_KEYS = { source: 1, label: 1, started: 1, pair_key: 1, server_summary: 1 };

  function getExportData() {
    if (!EXPORT) { return null; }
    if (exportData) { return exportData; }
    var node = document.getElementById('export-data');
    if (!node) { return null; }
    try {
      exportData = JSON.parse(node.textContent);
    } catch (e) {
      exportData = null;
    }
    return exportData;
  }

  function api(path, callback) {
    if (EXPORT) {
      var data = getExportData();
      if (!data) { callback(new Error('export data missing'), null); return; }
      if (path.indexOf('/api/state') === 0) {
        var q = path.indexOf('track=');
        var track = q < 0 ? 'val' : path.substring(q + 6);
        var states = data.states || {};
        var st = states[track];
        if (st) { callback(null, st); } else { callback(new Error('state not found for ' + track), null); }
        return;
      }
      if (path.indexOf('/api/run/') === 0) {
        var runId = decodeURIComponent(path.substring(9));
        var runs = data.runs || {};
        var detail = runs[runId];
        if (detail) { callback(null, detail); } else { callback(new Error('run not found: ' + runId), null); }
        return;
      }
      callback(new Error('unsupported path'), null);
      return;
    }
    var xhr = new XMLHttpRequest();
    xhr.open('GET', path, true);
    xhr.onreadystatechange = function () {
      if (xhr.readyState !== 4) { return; }
      if (xhr.status >= 200 && xhr.status < 300) {
        var parsed = null;
        try {
          parsed = JSON.parse(xhr.responseText);
        } catch (e) {
          callback(e, null);
          return;
        }
        callback(null, parsed);
      } else {
        callback(new Error('HTTP ' + xhr.status), null);
      }
    };
    xhr.onerror = function () { callback(new Error('network error'), null); };
    xhr.send();
  }

  function esc(value) {
    if (value === null || value === undefined) { return ''; }
    return String(value)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function num(value) {
    if (value === null || value === undefined || value === '') { return null; }
    var n = Number(value);
    if (isFinite(n)) { return n; }
    return null;
  }

  function fmt(value, digits) {
    var n = num(value);
    if (n === null) { return 'n/a'; }
    return n.toFixed(digits);
  }

  function fmtWith(value, digits, suffix) {
    var n = num(value);
    if (n === null) { return 'n/a'; }
    return n.toFixed(digits) + suffix;
  }

  function fmtInt(value) {
    var n = num(value);
    if (n === null) { return 'n/a'; }
    return String(Math.round(n));
  }

  function cellCount(value) {
    var n = num(value);
    if (n === null) { return ''; }
    return String(Math.round(n));
  }

  function fmtFracPct(value) {
    var n = num(value);
    if (n === null) { return 'n/a'; }
    return (n * 100).toFixed(1) + '%';
  }

  function fmtDuration(value) {
    var n = num(value);
    if (n === null) { return 'n/a'; }
    n = Math.max(0, Math.round(n));
    var hours = Math.floor(n / 3600);
    var mins = Math.floor((n % 3600) / 60);
    var secs = n % 60;
    if (hours > 0) { return hours + 'h ' + mins + 'm ' + secs + 's'; }
    if (mins > 0) { return mins + 'm ' + secs + 's'; }
    return secs + 's';
  }

  function fmtClock(value) {
    if (!value) { return '--:--:--'; }
    var s = String(value);
    if (s.length >= 19 && s.charAt(10) === 'T') { return s.substring(11, 19); }
    if (s.length >= 8) { return s.substring(s.length - 8); }
    return s;
  }

  function fmtStarted(value) {
    if (!value) { return 'n/a'; }
    var s = String(value);
    if (s.length >= 19 && s.charAt(10) === 'T') {
      return s.substring(5, 10) + ' ' + s.substring(11, 19);
    }
    return s;
  }

  function sloText(row) {
    var met = num(row.slo_met);
    var total = num(row.slo_total);
    if (met === null || total === null) { return 'n/a'; }
    return Math.round(met) + '/' + Math.round(total);
  }

  function fmtMetric(name, value) {
    var n = num(value);
    if (n === null) { return 'n/a'; }
    if (name === 'j_per_req') { return n.toFixed(0); }
    if (name === 'slo_met_frac') { return (n * 100).toFixed(1) + '%'; }
    if (name === 'p95_ttft_s') { return n.toFixed(2); }
    if (name === 'p95_tpot_ms') { return n.toFixed(1); }
    if (name === 'accuracy_pct') { return n.toFixed(1); }
    if (name === 'eta_vs_base') { return n.toFixed(3); }
    return String(value);
  }

  function sortValue(row, key) {
    if (key === 'slo_frac') {
      var met = num(row.slo_met);
      var total = num(row.slo_total);
      if (met === null || total === null || total === 0) { return null; }
      return met / total;
    }
    if (key === 'is_baseline') { return row.is_baseline ? 1 : 0; }
    var value = row[key];
    if (value === null || value === undefined || value === '') { return null; }
    if (TEXT_KEYS[key]) { return String(value); }
    return num(value);
  }

  function sortRows(rows) {
    var out = rows.slice();
    if (!sortKey) { return out; }
    var key = sortKey;
    var dir = sortDir;
    out.sort(function (a, b) {
      var av = sortValue(a, key);
      var bv = sortValue(b, key);
      if (av === null && bv === null) { return 0; }
      if (av === null) { return 1; }
      if (bv === null) { return -1; }
      if (typeof av === 'number' && typeof bv === 'number') { return (av - bv) * dir; }
      var as = String(av).toLowerCase();
      var bs = String(bv).toLowerCase();
      if (as < bs) { return -1 * dir; }
      if (as > bs) { return 1 * dir; }
      return 0;
    });
    return out;
  }

  function findRow(key) {
    if (!stateData || !stateData.history) { return null; }
    for (var i = 0; i < stateData.history.length; i += 1) {
      if (stateData.history[i].key === key) { return stateData.history[i]; }
    }
    return null;
  }

  function metricRow(label, value) {
    return '<div class="metric-row"><span>' + esc(label) + '</span><span>' + esc(value) + '</span></div>';
  }

  function baselineBox(title, row) {
    var html = '<div class="baseline-box"><div class="box-title">' + esc(title) + '</div>';
    if (!row) { return html + '<div class="muted">none</div></div>'; }
    html += '<div class="baseline-label">' + esc(row.label || row.id || '') + '</div>';
    html += metricRow('η/base', fmt(row.eta_vs_base, 3));
    html += metricRow('accuracy', fmtWith(row.accuracy_pct, 1, '%'));
    html += metricRow('S', fmt(row.S, 3));
    return html + '</div>';
  }

  function renderStatus() {
    var el = document.getElementById('status');
    if (!el) { return; }
    var data = stateData || {};
    var progress = data.progress;
    var html = '<div class="progress-block">';
    if (progress) {
      var total = num(progress.items_total);
      var done = num(progress.items_done);
      var max = total === null || total <= 0 ? 1 : total;
      var value = done === null ? 0 : Math.min(Math.max(done, 0), max);
      html += '<div class="progress-line"><strong>Run in progress:</strong> ';
      html += esc(progress.label || progress.run_id || '');
      html += ' — ' + esc(fmtInt(done)) + '/' + esc(fmtInt(total)) + ' items, ';
      html += esc(fmtDuration(progress.elapsed_s)) + ' elapsed';
      if (progress.stale) { html += ' <span class="stale">(stale)</span>'; }
      html += '</div>';
      html += '<progress max="' + esc(max) + '" value="' + esc(value) + '"></progress>';
    } else {
      html += '<div class="progress-line muted">No run in progress</div>';
    }
    html += '</div>';
    var baselines = data.baselines || {};
    html += baselineBox('Local baseline', baselines.local);
    html += baselineBox('Official baseline', baselines.official);
    el.innerHTML = html;
  }

  function bestCard(title, row, side) {
    var cls = side === 'official' ? 'card best-card official' : 'card best-card';
    var html = '<div class="' + cls + '"><div class="card-title">' + esc(title) + '</div>';
    if (!row) { return html + '<div class="best-value muted">n/a</div></div>'; }
    html += '<div class="best-value">' + esc(fmt(row.S, 1)) + '</div>';
    html += '<div class="best-label">' + esc(row.label || row.id || '') + '</div>';
    html += '<div class="best-server">' + esc(row.server_summary || row.flags || '') + '</div>';
    return html + '</div>';
  }

  function renderBest() {
    var el = document.getElementById('best');
    if (!el) { return; }
    var best = stateData && stateData.best ? stateData.best : {};
    el.innerHTML = bestCard('Best local S', best.local, 'local') + bestCard('Best official S', best.official, 'official');
  }

  function rowHtml(row) {
    var classes = 'row-' + (row.source === 'official' ? 'official' : 'local');
    if (row.is_baseline) { classes += ' baseline'; }
    if (row.key === selectedKey) { classes += ' selected'; }
    var label = esc(row.label || '');
    if (row.is_baseline) { label += ' <span class="badge badge-base">baseline</span>'; }
    if (row.source === 'local' && row.status && row.status !== 'done') {
      label += ' <span class="status muted">' + esc(row.status) + '</span>';
    }
    var eta = esc(fmt(row.eta_vs_base, 3));
    if (row.within_noise === true) { eta += ' <span class="badge badge-noise">≈ noise</span>'; }
    var pair = row.pair_key ? String(row.pair_key) : '';
    var config = row.server_summary || row.flags || '';
    var html = '<tr class="' + classes + '" data-key="' + esc(row.key) + '">';
    html += '<td class="src">' + esc(row.source) + '</td>';
    html += '<td class="label-cell">' + label + '</td>';
    html += '<td class="mono" title="' + esc(row.started) + '">' + esc(fmtStarted(row.started)) + '</td>';
    html += '<td class="num">' + esc(fmt(row.S, 3)) + '</td>';
    html += '<td class="num">' + esc(fmt(row.A, 3)) + '</td>';
    html += '<td class="num">' + esc(fmt(row.R, 3)) + '</td>';
    html += '<td class="num">' + esc(fmt(row.s_E, 3)) + '</td>';
    html += '<td class="num">' + esc(fmt(row.s_T, 3)) + '</td>';
    html += '<td class="num">' + esc(fmt(row.s_L, 3)) + '</td>';
    html += '<td class="num">' + eta + '</td>';
    html += '<td class="num">' + esc(fmt(row.j_per_req, 0)) + '</td>';
    html += '<td class="num">' + esc(fmt(row.p95_ttft_s, 2)) + '</td>';
    html += '<td class="num">' + esc(fmt(row.p95_tpot_ms, 1)) + '</td>';
    html += '<td class="num">' + esc(sloText(row)) + '</td>';
    html += '<td class="num">' + esc(fmtWith(row.accuracy_pct, 1, '%')) + '</td>';
    html += '<td>' + (row.is_baseline ? 'yes' : '') + '</td>';
    html += '<td class="pair" title="' + esc(pair) + '">' + esc(pair) + '</td>';
    html += '<td class="config" title="' + esc(config) + '">' + esc(config) + '</td>';
    return html + '</tr>';
  }

  function renderGroups() {
    var el = document.getElementById('groups');
    if (!el) { return; }
    var groups = stateData && stateData.groups ? stateData.groups : [];
    if (!groups.length) {
      el.innerHTML = '<div class="muted">No repeat groups.</div>';
      return;
    }
    var html = '<div class="groups-title">Repeat groups</div>';
    for (var i = 0; i < groups.length; i += 1) {
      var g = groups[i] || {};
      var mean = g.mean || {};
      var low = g.min || {};
      var high = g.max || {};
      html += '<div class="group-line">' + esc(g.group || 'group') + ' ×' + esc(fmtInt(g.n));
      html += ': η/base ' + esc(fmt(mean.eta_vs_base, 3));
      html += ' (' + esc(fmt(low.eta_vs_base, 3)) + '–' + esc(fmt(high.eta_vs_base, 3)) + ')</div>';
    }
    el.innerHTML = html;
  }

  function renderNoise() {
    var el = document.getElementById('noise');
    if (!el) { return; }
    var noise = stateData && stateData.noise ? stateData.noise : {};
    var spread = num(noise.spread);
    var pct = spread === null ? 'n/a' : (spread * 100).toFixed(1) + '%';
    var source = noise.observed ? 'observed from repeats' : 'default prior';
    el.innerHTML = 'Noise spread ±' + esc(pct) + ' (' + esc(source) + ')';
  }

  function updateSortHeaders() {
    var ths = document.querySelectorAll('#history th[data-sort]');
    for (var i = 0; i < ths.length; i += 1) {
      var key = ths[i].getAttribute('data-sort');
      if (key === sortKey) {
        ths[i].setAttribute('data-dir', sortDir === 1 ? 'asc' : 'desc');
      } else {
        ths[i].removeAttribute('data-dir');
      }
    }
  }

  function bindHistory() {
    var body = document.getElementById('history-body');
    if (body) {
      body.onclick = function (ev) {
        var node = ev.target;
        while (node && node !== body && node.tagName !== 'TR') { node = node.parentNode; }
        if (!node || node === body || node.tagName !== 'TR') { return; }
        var row = findRow(node.getAttribute('data-key'));
        if (row) { selectRow(row); }
      };
    }
    var ths = document.querySelectorAll('#history th[data-sort]');
    for (var i = 0; i < ths.length; i += 1) {
      ths[i].onclick = function () { toggleSort(this.getAttribute('data-sort')); };
    }
  }

  function renderHistory() {
    var body = document.getElementById('history-body');
    if (!body || !stateData) { return; }
    var rows = stateData.history || [];
    var sorted = sortRows(rows);
    var html = '';
    for (var i = 0; i < sorted.length; i += 1) { html += rowHtml(sorted[i]); }
    if (sorted.length === 0) {
      html = '<tr><td colspan="18" class="muted">No runs yet.</td></tr>';
    }
    body.innerHTML = html;
    bindHistory();
    updateSortHeaders();
    renderGroups();
    renderNoise();
  }

  function toggleSort(key) {
    if (!key) { return; }
    if (sortKey === key) {
      sortDir = -sortDir;
    } else {
      sortKey = key;
      sortDir = 1;
    }
    renderHistory();
  }

  function flagPill(flag) {
    var f = flag ? String(flag) : 'na';
    var cls = 'pill';
    if (f === 'green' || f === 'amber' || f === 'red') {
      cls += ' pill-' + f;
    } else {
      cls += ' pill-na';
    }
    return '<span class="' + cls + '">' + esc(f) + '</span>';
  }

  function pairHtml(pair) {
    var html = '<div class="cal-pair"><h3>' + esc(pair.label || 'pair') + '</h3>';
    html += '<table class="grid cal-table"><thead><tr>';
    html += '<th>Metric</th><th class="num">Local</th><th class="num">Official</th><th class="num">Ratio</th><th>Flag</th>';
    html += '</tr></thead><tbody>';
    var metrics = pair.metrics || [];
    for (var i = 0; i < metrics.length; i += 1) {
      var m = metrics[i] || {};
      html += '<tr>';
      html += '<td>' + esc(m.label || m.name || '') + '</td>';
      html += '<td class="num">' + esc(fmtMetric(m.name, m.local)) + '</td>';
      html += '<td class="num">' + esc(fmtMetric(m.name, m.official)) + '</td>';
      html += '<td class="num">' + esc(fmt(m.ratio, 3)) + '</td>';
      html += '<td>' + flagPill(m.flag) + '</td>';
      html += '</tr>';
    }
    return html + '</tbody></table></div>';
  }

  function renderCalibration() {
    var el = document.getElementById('calibration');
    if (!el) { return; }
    var cal = stateData && stateData.calibration ? stateData.calibration : {};
    var pairs = cal.pairs || [];
    var html = '<h2>Calibration</h2>';
    if (!pairs.length) {
      html += '<p class="muted">No linked local/official pairs yet — use: grader.py official add … --local RUN_ID</p>';
    } else {
      for (var i = 0; i < pairs.length; i += 1) { html += pairHtml(pairs[i] || {}); }
    }
    var rank = cal.rank_agreement || {};
    if (rank.text) { html += '<p class="rank-line">' + esc(rank.text) + '</p>'; }
    el.innerHTML = html;
  }

  function setChart(id, svg) {
    var el = document.getElementById(id);
    if (!el) { return; }
    if (typeof svg === 'string' && svg.length > 0) {
      el.innerHTML = svg;
    } else {
      el.innerHTML = '<span class="muted">no data</span>';
    }
  }

  function renderCharts() {
    var charts = stateData && stateData.charts ? stateData.charts : {};
    setChart('chart-s-over-time', charts.s_over_time);
    setChart('chart-scatter', charts.scatter);
    setChart('chart-eta-bars', charts.eta_bars);
  }

  function showDetailMessage(text) {
    var el = document.getElementById('detail');
    if (!el) { return; }
    el.innerHTML = '<h2>Detail</h2><p class="muted">' + esc(text) + '</p>';
  }

  function classRowsHtml(rows) {
    if (!rows || !rows.length) {
      return '<h3>Classes</h3><p class="muted">No class data.</p>';
    }
    var html = '<h3>Classes</h3><table class="grid detail-table"><thead><tr>';
    html += '<th>Class</th><th class="num">Acc</th><th class="num">n</th>';
    html += '<th class="num">Baseline acc</th><th class="num">Ratio</th><th>Flag</th>';
    html += '</tr></thead><tbody>';
    for (var i = 0; i < rows.length; i += 1) {
      var c = rows[i] || {};
      var trClass = c.flag === 'low' ? ' class="row-low"' : '';
      var flag = c.flag === 'low'
        ? '<span class="low-note">below 0.75 of baseline</span>'
        : esc(c.flag || 'ok');
      html += '<tr' + trClass + '>';
      html += '<td>' + esc(c.cls || '') + '</td>';
      html += '<td class="num">' + esc(fmtFracPct(c.acc)) + '</td>';
      html += '<td class="num">' + esc(fmtInt(c.n)) + '</td>';
      html += '<td class="num">' + esc(fmtFracPct(c.base_acc)) + '</td>';
      html += '<td class="num">' + esc(fmt(c.ratio, 3)) + '</td>';
      html += '<td>' + flag + '</td>';
      html += '</tr>';
    }
    return html + '</tbody></table>';
  }

  function indexOfText(arr, value) {
    for (var i = 0; i < arr.length; i += 1) {
      if (arr[i] === value) { return i; }
    }
    return -1;
  }

  function failTableHtml(fail) {
    if (!fail) {
      return '<h3>Failures by phase</h3><p class="muted">No failure data.</p>';
    }
    var phases = [];
    var reasons = [];
    var ph;
    var reason;
    for (ph in fail) {
      if (!Object.prototype.hasOwnProperty.call(fail, ph)) { continue; }
      phases.push(ph);
      var counts = fail[ph] || {};
      for (reason in counts) {
        if (!Object.prototype.hasOwnProperty.call(counts, reason)) { continue; }
        if (indexOfText(reasons, reason) < 0) { reasons.push(reason); }
      }
    }
    if (!phases.length) {
      return '<h3>Failures by phase</h3><p class="muted">No failure data.</p>';
    }
    phases.sort();
    reasons.sort();
    var html = '<h3>Failures by phase</h3><table class="grid detail-table"><thead><tr><th>Phase</th>';
    for (var i = 0; i < reasons.length; i += 1) {
      html += '<th class="num">' + esc(reasons[i]) + '</th>';
    }
    html += '</tr></thead><tbody>';
    for (var p = 0; p < phases.length; p += 1) {
      var row = fail[phases[p]] || {};
      html += '<tr><td>' + esc(phases[p]) + '</td>';
      for (var j = 0; j < reasons.length; j += 1) {
        html += '<td class="num">' + esc(cellCount(row[reasons[j]])) + '</td>';
      }
      html += '</tr>';
    }
    return html + '</tbody></table>';
  }

  function tokensHtml(tokens) {
    if (!tokens) {
      return '<h3>Tokens by class</h3><p class="muted">No token data.</p>';
    }
    var classes = [];
    var cls;
    for (cls in tokens) {
      if (!Object.prototype.hasOwnProperty.call(tokens, cls)) { continue; }
      classes.push(cls);
    }
    if (!classes.length) {
      return '<h3>Tokens by class</h3><p class="muted">No token data.</p>';
    }
    classes.sort();
    var html = '<h3>Tokens by class</h3><table class="grid detail-table"><thead><tr>';
    html += '<th>Class</th><th class="num">Mean</th><th class="num">p95</th></tr></thead><tbody>';
    for (var i = 0; i < classes.length; i += 1) {
      var t = tokens[classes[i]] || {};
      html += '<tr><td>' + esc(classes[i]) + '</td>';
      html += '<td class="num">' + esc(fmt(t.mean, 1)) + '</td>';
      html += '<td class="num">' + esc(fmt(t.p95, 1)) + '</td></tr>';
    }
    return html + '</tbody></table>';
  }

  function renderLocalDetail(detail, stateRow) {
    var el = document.getElementById('detail');
    if (!el) { return; }
    var d = detail || {};
    var r = d.row || stateRow || {};
    var html = '<h2>' + esc(r.label || r.id || 'Run');
    html += ' <span class="mono muted">' + esc(r.id || '') + '</span></h2>';
    var links = '';
    if (!EXPORT && d.report_url && String(d.report_url).charAt(0) === '/') {
      links += '<a class="report-link" target="_blank" rel="noopener noreferrer" href="' + esc(d.report_url) + '">Open report</a>';
    }
    if (r.status && r.status !== 'done') {
      links += ' <span class="muted">status: ' + esc(r.status) + '</span>';
    }
    if (links) { html += '<p class="detail-links">' + links + '</p>'; }
    if (r.server_summary || r.flags) {
      html += '<p class="server-summary">' + esc(r.server_summary || r.flags) + '</p>';
    }
    var facts = [];
    facts.push(['S', fmt(r.S, 3)]);
    facts.push(['η/base', fmt(r.eta_vs_base, 3)]);
    facts.push(['J/req', fmt(r.j_per_req, 0)]);
    facts.push(['SLO', sloText(r)]);
    facts.push(['Acc', fmtWith(r.accuracy_pct, 1, '%')]);
    html += '<div class="facts">';
    for (var i = 0; i < facts.length; i += 1) {
      html += '<div class="fact"><span class="fact-k">' + esc(facts[i][0]) + '</span>';
      html += '<span class="fact-v">' + esc(facts[i][1]) + '</span></div>';
    }
    html += '</div>';
    html += classRowsHtml(d.class_rows);
    html += failTableHtml(d.fail_by_phase);
    html += tokensHtml(d.tokens_by_class);
    html += '<h3>Power</h3><div id="detail-power" class="chart-box"></div>';
    el.innerHTML = html;
    setChart('detail-power', d.power_svg);
  }

  function officialFact(label, value) {
    return '<tr><th>' + esc(label) + '</th><td class="num">' + esc(value) + '</td></tr>';
  }

  function renderOfficialDetail(row) {
    var el = document.getElementById('detail');
    if (!el) { return; }
    var html = '<h2>Official: ' + esc(row.label || '');
    html += ' <span class="mono muted">#' + esc(row.id) + '</span></h2>';
    if (row.server_summary || row.flags) {
      html += '<p class="server-summary">' + esc(row.server_summary || row.flags) + '</p>';
    }
    html += '<table class="grid detail-table"><tbody>';
    html += officialFact('Track', row.track || '');
    html += officialFact('Started', fmtStarted(row.started));
    html += officialFact('S', fmt(row.S, 3));
    html += officialFact('A', fmt(row.A, 3));
    html += officialFact('R', fmt(row.R, 3));
    html += officialFact('sE', fmt(row.s_E, 3));
    html += officialFact('sT', fmt(row.s_T, 3));
    html += officialFact('sL', fmt(row.s_L, 3));
    html += officialFact('η/base', fmt(row.eta_vs_base, 3));
    html += officialFact('J/req', fmt(row.j_per_req, 0));
    html += officialFact('p95 TTFT (s)', fmt(row.p95_ttft_s, 2));
    html += officialFact('p95 TPOT (ms)', fmt(row.p95_tpot_ms, 1));
    html += officialFact('SLO', sloText(row));
    html += officialFact('Acc %', fmtWith(row.accuracy_pct, 1, '%'));
    html += officialFact('Pair', row.pair_key || '');
    html += '</tbody></table>';
    if (row.notes) {
      html += '<h3>Notes</h3><p class="notes">' + esc(row.notes) + '</p>';
    } else {
      html += '<h3>Notes</h3><p class="muted">No notes.</p>';
    }
    el.innerHTML = html;
  }

  function renderSelection() {
    if (!selectedKey) {
      showDetailMessage('Select a run in the history table.');
      return;
    }
    var row = findRow(selectedKey);
    if (!row) {
      showDetailMessage('The selected run is not in this track.');
      return;
    }
    if (row.source === 'official') {
      renderOfficialDetail(row);
      return;
    }
    if (detailCache[selectedKey]) {
      renderLocalDetail(detailCache[selectedKey], row);
    } else {
      showDetailMessage('Select a run in the history table.');
    }
  }

  function selectRow(row) {
    selectedKey = row.key;
    renderHistory();
    if (row.source === 'official') {
      renderOfficialDetail(row);
      return;
    }
    showDetailMessage('Loading run ' + row.id + ' …');
    api('/api/run/' + encodeURIComponent(row.id), function (err, detail) {
      if (selectedKey !== row.key) { return; }
      if (err) {
        showDetailMessage('Could not load run: ' + (err.message || String(err)));
        return;
      }
      detailCache[row.key] = detail;
      renderLocalDetail(detail, row);
    });
  }

  function renderUpdated() {
    var el = document.getElementById('updated');
    if (!el || !stateData) { return; }
    el.textContent = 'updated ' + fmtClock(stateData.generated_at);
  }

  function renderAll() {
    if (!stateData) { return; }
    renderUpdated();
    renderStatus();
    renderBest();
    renderHistory();
    renderCalibration();
    renderCharts();
    renderSelection();
  }

  function loadState() {
    api('/api/state?track=' + TRACK, function (err, data) {
      if (err) {
        var el = document.getElementById('status');
        if (el) {
          el.innerHTML = '<span class="error">Could not load state: ' + esc(err.message || String(err)) + '</span>';
        }
        return;
      }
      stateData = data || {};
      renderAll();
    });
  }

  function readTrack() {
    var h = String(location.hash || '').replace('#', '');
    return h === 'final' ? 'final' : 'val';
  }

  function updateTrackButtons() {
    var buttons = [['track-val', 'val'], ['track-final', 'final']];
    for (var i = 0; i < buttons.length; i += 1) {
      var el = document.getElementById(buttons[i][0]);
      if (!el) { continue; }
      el.className = buttons[i][1] === TRACK ? 'track-btn active' : 'track-btn';
    }
  }

  function selectTrack(track) {
    var next = track === 'final' ? 'final' : 'val';
    if (next === TRACK) { return; }
    TRACK = next;
    selectedKey = null;
    detailCache = {};
    if (location.hash !== '#' + TRACK) { location.hash = TRACK; }
    updateTrackButtons();
    showDetailMessage('Select a run in the history table.');
    loadState();
  }

  function init() {
    TRACK = readTrack();
    updateTrackButtons();
    var valButton = document.getElementById('track-val');
    var finalButton = document.getElementById('track-final');
    if (valButton) { valButton.onclick = function () { selectTrack('val'); }; }
    if (finalButton) { finalButton.onclick = function () { selectTrack('final'); }; }
    window.onhashchange = function () {
      var track = readTrack();
      if (track !== TRACK) { selectTrack(track); }
    };
    showDetailMessage('Select a run in the history table.');
    loadState();
    if (!EXPORT) {
      setInterval(function () { loadState(); }, 15000);
    }
  }

  init();
})();
</script>
</body>
</html>
"""


def render_page(export=None):
    """Return the live dashboard, or a static snapshot with embedded JSON data."""
    if export is None:
        return PAGE_HTML
    payload = export if isinstance(export, dict) else {}
    raw = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    raw = raw.replace("<!--", "<\\u0021--").replace("</", "<\\/")
    data_tag = (
        '<script id="export-data" type="application/json">'
        + raw
        + "</script>\n"
    )
    return _TEMPLATE.replace("@@EXPORT_FLAG@@", "true").replace("@@EXPORT_DATA@@", data_tag)


PAGE_HTML = _TEMPLATE.replace("@@EXPORT_FLAG@@", "false").replace("@@EXPORT_DATA@@", "")
