"""Self-contained SVG charts for the phase-2 dashboard.

Every public function is pure, takes flat history ROW dicts (see
INTERFACES2.md section 3) and returns a single ``<svg ...>...</svg>``
string.  No JavaScript, no external references, no network access.
All colours go through CSS custom properties with hard-coded hex
fallbacks (e.g. ``var(--fg, #111827)``) so the page theme controls the
look.  Every piece of user-provided text passes through html.escape.
"""

from __future__ import annotations

import html
import math
from datetime import datetime

__all__ = ["s_over_time", "scatter_j_vs_slo", "eta_bars", "power_chart"]

# CSS variables with fallbacks -- the page theme overrides these.
_LOCAL = "var(--local, #2563eb)"
_OFFICIAL = "var(--official, #d97706)"
_FG = "var(--fg, #111827)"
_GRID = "var(--grid, #d1d5db)"
_ACCENT = "var(--accent, #059669)"

# Phase bands alternate between two light fills (fill-opacity 0.12).
_BAND_A = _GRID
_BAND_B = _ACCENT
_BAND_OPACITY = "0.12"


# ---------------------------------------------------------------- helpers

def _num(value):
    """float(value) or None; never raises, NaN/inf filtered."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _esc(text):
    """html-escape anything for use in element text or attributes."""
    return html.escape(str(text), quote=True)


def _tick_label(value):
    """Trim a float to a short tick label: 0.75 -> '0.75', 350.0 -> '350'."""
    s = f"{value:.2f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-") else "0"


def _text(x, y, body, anchor="start", size=11, fill=_FG, transform=None):
    extra = f' transform="{_esc(transform)}"' if transform else ""
    return (
        f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}"'
        f' font-size="{size:g}" fill="{fill}"{extra}>{_esc(body)}</text>'
    )


def _grid_line(x1, y1, x2, y2):
    return (
        f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"'
        f' stroke="{_GRID}" stroke-width="1"/>'
    )


def _no_data():
    """Small placeholder svg (still a valid, self-contained chart)."""
    inner = _text(100, 38, "no data", anchor="middle", size=12)
    return _wrap(inner, 200, 64, "no data")


def _wrap(inner, width, height, label):
    return (
        f'<svg viewBox="0 0 {width:g} {height:g}" width="100%"'
        f' style="max-width:{width:g}px" role="img"'
        f' aria-label="{_esc(label)}" font-family="inherit">{inner}</svg>'
    )


def _source_fill(row):
    return _OFFICIAL if row.get("source") == "official" else _LOCAL


def _diamond(cx, cy, half, row, stroke_extra=""):
    """Rotated square centred on (cx, cy)."""
    return (
        f'<rect x="{cx - half:g}" y="{cy - half:g}"'
        f' width="{2 * half:g}" height="{2 * half:g}"'
        f' fill="{_source_fill(row)}"{stroke_extra}'
        f' transform="rotate(45 {cx:g} {cy:g})"/>'
    )


def _circle(cx, cy, r, row, stroke_extra=""):
    return (
        f'<circle cx="{cx:g}" cy="{cy:g}" r="{r:g}"'
        f' fill="{_source_fill(row)}"{stroke_extra}/>'
    )


def _baseline_stroke(width):
    """Thicker outline for baseline rows."""
    return f' stroke="{_FG}" stroke-width="{width:g}"'


# ------------------------------------------------------------ s_over_time

def s_over_time(rows):
    """x = started time, y = S.  local = circles, official = diamonds."""
    pts = []
    for row in rows or []:
        s = _num(row.get("S"))
        if s is None:
            continue
        try:
            t = datetime.fromisoformat(str(row.get("started")))
        except (TypeError, ValueError):
            continue
        pts.append((t, s, row))
    if not pts:
        return _no_data()

    w, h = 720.0, 260.0
    ml, mr, mt, mb = 46.0, 16.0, 30.0, 34.0
    pw, ph = w - ml - mr, h - mt - mb

    if len(pts) == 1:
        # Exactly one plottable point: centre it.
        t, s, row = pts[0]
        stroke = _baseline_stroke(2.5) if row.get("is_baseline") else ""
        inner = _circle(ml + pw / 2, mt + ph / 2, 4, row, stroke)
        if row.get("source") == "official":
            inner = _diamond(ml + pw / 2, mt + ph / 2, 4.5, row, stroke)
        inner += _text(
            ml + pw / 2, mt + ph / 2 + 18,
            f"{t.strftime('%m-%d %H:%M')}  S={_tick_label(s)}",
            anchor="middle", size=11,
        )
        return _wrap(inner, w, h, "Score S over start time (single point)")

    t0 = min(p[0] for p in pts)
    t1 = max(p[0] for p in pts)
    span = (t1 - t0).total_seconds()
    ymax = max(1.0, max(p[1] for p in pts))

    def px(t):
        if span <= 0:
            return ml + pw / 2
        return ml + (t - t0).total_seconds() / span * pw

    def py(s):
        return mt + (1.0 - s / ymax) * ph

    parts = []

    # y grid lines + tick labels
    for i in range(5):
        value = ymax * i / 4
        y = py(value)
        parts.append(_grid_line(ml, y, w - mr, y))
        parts.append(_text(ml - 5, y + 3.5, _tick_label(value), anchor="end"))

    # axes
    parts.append(_grid_line(ml, mt, ml, h - mb))
    parts.append(_grid_line(ml, h - mb, w - mr, h - mb))

    # x time ticks (up to 5, 'MM-DD HH:MM')
    for i in range(5):
        t = t0 + (t1 - t0) * i / 4
        parts.append(_text(px(t), h - 10, t.strftime("%m-%d %H:%M"),
                           anchor="middle"))

    # data points
    for t, s, row in pts:
        cx, cy = px(t), py(s)
        stroke = _baseline_stroke(2.5) if row.get("is_baseline") else ""
        if row.get("source") == "official":
            parts.append(_diamond(cx, cy, 4.5, row, stroke))
        else:
            parts.append(_circle(cx, cy, 3.5, row, stroke))

    # legend: Local / Official
    lx = w - mr - 132.0
    parts.append(_circle(lx, 14, 3.5, {"source": "local"}))
    parts.append(_text(lx + 8, 18, "Local"))
    parts.append(_diamond(lx + 64, 14, 4, {"source": "official"}))
    parts.append(_text(lx + 71, 18, "Official"))

    return _wrap(
        "".join(parts), w, h,
        "Score S over start time; circles are local runs, diamonds are "
        "official runs, outlined marks are baselines",
    )


# -------------------------------------------------------- scatter_j_vs_slo

def scatter_j_vs_slo(rows):
    """x = j_per_req, y = slo_met / slo_total (fixed 0..1), one dot per row."""
    pts = []
    for row in rows or []:
        j = _num(row.get("j_per_req"))
        met = _num(row.get("slo_met"))
        total = _num(row.get("slo_total"))
        if j is None or met is None or total is None or total <= 0:
            continue
        pts.append((j, met / total, row))
    if not pts:
        return _no_data()

    w, h = 720.0, 320.0
    ml, mr, mt, mb = 48.0, 12.0, 26.0, 42.0
    pw, ph = w - ml - mr, h - mt - mb

    xs = [p[0] for p in pts]
    xmin, xmax = min(xs), max(xs)
    if xmax - xmin <= 0:
        pad = max(abs(xmax) * 0.5, 1.0)
    else:
        pad = (xmax - xmin) * 0.05
    xmin, xmax = xmin - pad, xmax + pad

    def px(j):
        return ml + (j - xmin) / (xmax - xmin) * pw

    def py(y):
        return mt + (1.0 - y) * ph

    parts = []

    # y grid lines + tick labels (0 .. 1)
    for i in range(5):
        value = i / 4
        y = py(value)
        parts.append(_grid_line(ml, y, w - mr, y))
        parts.append(_text(ml - 5, y + 3.5, _tick_label(value), anchor="end"))

    # axes
    parts.append(_grid_line(ml, mt, ml, h - mb))
    parts.append(_grid_line(ml, h - mb, w - mr, h - mb))

    # x ticks
    for i in range(5):
        value = xmin + (xmax - xmin) * i / 4
        parts.append(_text(px(value), h - mb + 16, _tick_label(value),
                           anchor="middle"))

    # dots + labels
    for j, y, row in pts:
        cx, cy = px(j), py(y)
        stroke = _baseline_stroke(2.5) if row.get("is_baseline") else ""
        parts.append(_circle(cx, cy, 4, row, stroke))
        label = str(row.get("label") or row.get("id") or "")[:18]
        if label:
            if (j - xmin) / (xmax - xmin) >= 0.7:
                # right 30%: anchor end so the text cannot overflow
                parts.append(_text(cx - 6, cy + 4, label, anchor="end"))
            else:
                parts.append(_text(cx + 6, cy + 4, label))

    # corner note + axis titles
    parts.append(_text(ml + 8, mt + 14, "upper-left = better"))
    parts.append(_text(ml + pw / 2, h - 8, "J / request", anchor="middle"))
    parts.append(_text(
        13, mt + ph / 2, "SLO met", anchor="middle",
        transform=f"rotate(-90 13 {mt + ph / 2:g})",
    ))

    return _wrap(
        "".join(parts), w, h,
        "Joules per request versus SLO met fraction; upper-left is better",
    )


# ---------------------------------------------------------------- eta_bars

def eta_bars(rows):
    """Horizontal bars of eta_vs_base (newest first, max 25 rows)."""
    selected = []
    for row in rows or []:
        value = _num(row.get("eta_vs_base"))
        if value is None:
            continue
        selected.append((value, row))
        if len(selected) >= 25:
            break
    if not selected:
        return _no_data()

    w = 680.0
    row_h, bar_h = 22.0, 12.0
    ml, mr, mt, mb = 210.0, 64.0, 26.0, 14.0
    n = len(selected)
    h = mt + n * row_h + mb
    pw = w - ml - mr

    x_max = max(1.0, max(v for v, _ in selected)) * 1.06

    def sx(value):
        return ml + value / x_max * pw

    parts = []

    for i, (value, row) in enumerate(selected):
        top = mt + i * row_h
        mid = top + row_h / 2
        prefix = {"local": "L ", "official": "O "}.get(row.get("source"), "")
        label = prefix + str(row.get("label") or row.get("id") or "")[:28]

        bar_w = max(sx(value) - ml, 0.5)
        parts.append(
            f'<rect x="{ml:g}" y="{top + (row_h - bar_h) / 2:g}"'
            f' width="{bar_w:g}" height="{bar_h:g}"'
            f' fill="{_source_fill(row)}"/>'
        )
        parts.append(_text(ml - 6, mid + 4, label, anchor="end"))
        parts.append(_text(sx(value) + 5, mid + 4, f"{value:.3f}"))

    # dashed reference line at 1.0 labelled 'baseline'
    bx = sx(1.0)
    parts.append(
        f'<line x1="{bx:g}" y1="{mt:g}" x2="{bx:g}" y2="{h - mb:g}"'
        f' stroke="{_FG}" stroke-width="1" stroke-dasharray="4 3"/>'
    )
    parts.append(_text(bx, mt - 8, "baseline", anchor="middle"))

    return _wrap(
        "".join(parts), w, h,
        "Energy per request relative to baseline (eta_vs_base); "
        "dashed line marks 1.0",
    )


# ------------------------------------------------------------- power_chart

def power_chart(power, bounds, width=720, height=220):
    """Power trace [[t_s, watts], ...] with shaded phase bands."""
    if not power:
        return _no_data()
    pts = []
    for point in power:
        try:
            t, watts = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isnan(t) or math.isinf(t) or math.isnan(watts) or math.isinf(watts):
            continue
        pts.append((t, watts))
    if not pts:
        return _no_data()

    # downsample to <= 500 points by striding
    if len(pts) > 500:
        stride = math.ceil(len(pts) / 500)
        pts = pts[::stride]

    w, h = float(width), float(height)
    ml, mr, mt, mb = 52.0, 14.0, 26.0, 40.0
    pw, ph = w - ml - mr, h - mt - mb

    t0 = min(t for t, _ in pts)
    t1 = max(t for t, _ in pts)
    span = t1 - t0 if t1 > t0 else 1.0
    w_max = max(p for _, p in pts)
    if w_max <= 0:
        w_max = 1.0
    y_max = w_max * 1.08
    y_min = 0.0

    def sx(t):
        return ml + (t - t0) / span * pw

    def sy(p):
        return mt + (1.0 - (p - y_min) / (y_max - y_min)) * ph

    parts = []

    # shaded phase bands (alternating light fills)
    bands = []
    bounded = bounds if isinstance(bounds, dict) else {}
    for name in ("ramp", "peak", "quiet"):
        window = bounded.get(name)
        if isinstance(window, (list, tuple)) and len(window) == 2:
            bands.append((name, float(window[0]), float(window[1])))
    quiet = bounded.get("quiet")
    if isinstance(quiet, (list, tuple)) and len(quiet) == 2 and t1 > float(quiet[1]):
        bands.append(("drain", float(quiet[1]), t1))
    elif isinstance(bounded.get("drain"), (list, tuple)) \
            and len(bounded["drain"]) == 2:
        window = bounded["drain"]
        bands.append(("drain", float(window[0]), float(window[1])))

    for i, (name, a, b) in enumerate(bands):
        a = max(a, t0)
        b = min(b, t1)
        if b <= a:
            continue
        x1, x2 = sx(a), sx(b)
        fill = _BAND_A if i % 2 == 0 else _BAND_B
        parts.append(
            f'<rect x="{x1:g}" y="{mt:g}" width="{x2 - x1:g}"'
            f' height="{ph:g}" fill="{fill}" fill-opacity="{_BAND_OPACITY}"/>'
        )
        parts.append(_text((x1 + x2) / 2, mt + 13, name, anchor="middle"))

    # y grid lines + tick labels (watts)
    for value in (y_min, (y_min + y_max) / 2, y_max):
        y = sy(value)
        parts.append(_grid_line(ml, y, w - mr, y))
        parts.append(_text(ml - 5, y + 3.5, _tick_label(value), anchor="end"))

    # axes
    parts.append(_grid_line(ml, mt, ml, h - mb))
    parts.append(_grid_line(ml, h - mb, w - mr, h - mb))

    # x ticks (seconds, up to 6)
    for i in range(6):
        t = t0 + span * i / 5
        parts.append(_text(sx(t), h - mb + 16, _tick_label(t), anchor="middle"))

    # polyline of the trace
    points = " ".join(f"{sx(t):g},{sy(p):g}" for t, p in pts)
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="{_ACCENT}"'
        f' stroke-width="2"/>'
    )

    # axis labels
    parts.append(_text(ml + pw / 2, h - 6, "seconds", anchor="middle"))
    parts.append(_text(
        13, mt + ph / 2, "watts", anchor="middle",
        transform=f"rotate(-90 13 {mt + ph / 2:g})",
    ))

    return _wrap(
        "".join(parts), w, h,
        "Power draw in watts over time with shaded run phases "
        "(ramp, peak, quiet, drain)",
    )
