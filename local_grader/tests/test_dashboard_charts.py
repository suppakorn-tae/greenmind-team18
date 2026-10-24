"""Tests for gm/dashboard_charts.py (phase 2, agent C)."""

import re
import xml.etree.ElementTree as ET

import pytest

from gm.dashboard_charts import eta_bars, power_chart, s_over_time, scatter_j_vs_slo


# ------------------------------------------------------------------ helpers

def row(**kw):
    """A flat history ROW with sensible defaults; overrides via kwargs."""
    base = {
        "key": "local:run1",
        "source": "local",
        "id": "run1",
        "track": "val",
        "label": "run1",
        "started": "2026-10-24T10:00:00",
        "status": "done",
        "is_baseline": False,
        "S": 0.8,
        "eta_vs_base": 1.05,
        "j_per_req": 120.0,
        "slo_met": 9,
        "slo_total": 10,
    }
    base.update(kw)
    return base


USAGE = 'role="img"', 'aria-label=', 'viewBox=', 'width="100%"', 'max-width:'


def assert_svg(svg):
    """Structural contract shared by every chart function."""
    s = svg.strip()
    assert s.startswith("<svg"), s[:80]
    assert s.endswith("</svg>"), s[-80:]
    assert "http" not in s, "svg must not contain external references"
    assert "<script" not in s.lower(), "svg must not contain scripts"
    assert 'font-family="inherit"' in s
    assert "var(--" in s, "colours must go through CSS variables"
    for attr in USAGE:
        assert attr in s, f"missing {attr}"
    el = ET.fromstring(s)  # proves well-formed XML
    assert el.tag == "svg"
    assert el.get("viewBox")
    assert el.get("role") == "img"
    assert el.get("aria-label")
    assert el.get("width") == "100%"
    return el


def chart_texts(svg):
    """Concatenated text of all <text> elements (pytest Strawberry-free)."""
    el = ET.fromstring(svg.strip())
    return " ".join((t.text or "") for t in el.iter("text"))


# ------------------------------------------------------------- no data case

@pytest.mark.parametrize(
    "fn",
    [lambda: s_over_time([]),
     lambda: eta_bars([]),
     lambda: scatter_j_vs_slo([])])
def test_empty_gives_no_data(fn):
    svg = fn()
    assert_svg(svg)
    assert "no data" in chart_texts(svg)


def test_all_none_gives_no_data():
    rows = [
        row(S=None, eta_vs_base=None, j_per_req=None, slo_met=None, slo_total=None),
        row(key="local:r2", id="r2", S=None, eta_vs_base=None,
            j_per_req=None, slo_met=None, slo_total=None),
    ]
    for svg in (s_over_time(rows), scatter_j_vs_slo(rows), eta_bars(rows)):
        assert_svg(svg)
        assert "no data" in chart_texts(svg)


# --------------------------------------------------------------- s_over_time

def test_s_over_time_many_points():
    rows = [
        row(is_baseline=True,
            started="2026-10-24T09:00:00", S=0.0,
            eta_vs_base=1.0, j_per_req=100.0, slo_met=8, slo_total=10,
            label="base"),
        row(key="official:1", source="official", id=1, label="off1",
            started="2026-10-25T12:30:00", S=0.6),
        row(key="official:2", source="official", id=2, label="off2",
            started="2026-10-26T18:00:00", S=0.7),
        row(key="local:r2", id="r2", label="cand",
            started="2026-10-27T08:15:00", S=0.9),
    ]
    svg = s_over_time(rows)
    el = assert_svg(svg)
    texts = chart_texts(svg)
    # legend + escaped-free ticks
    assert "Local" in texts and "Official" in texts
    assert "<circle" in svg and "rotate(45" in svg  # local dot + official diamond
    # x tick label format MM-DD HH:MM
    assert re.search(r"\d{2}-\d{2} \d{2}:\d{2}", texts)
    assert el.get("viewBox").startswith("0 0 720 260")


def test_s_over_time_single_point_centred():
    svg = s_over_time([row(S=0.82)])
    el = assert_svg(svg)
    circle = next(e for e in el.iter("circle"))
    cx = float(circle.get("cx"))
    assert 200 < cx < 520  # roughly centred in the 720-wide viewBox


def test_s_over_time_skips_bad_rows():
    rows = [
        row(started="not-a-date", S=0.5),                       # unparseable time
        row(key="local:r2", id="r2", S="oops"),                 # non-numeric S
        row(key="local:r3", id="r3", S=None),                   # S None
        row(key="local:r4", id="r4", started="2026-10-24T11:00:00", S=0.5),
    ]
    assert_svg(s_over_time(rows))  # skips unusable rows, keeps the good one


# ---------------------------------------------------------- scatter_j_vs_slo

def test_scatter_points_and_labels():
    rows = [
        row(label="short", j_per_req=80.0, slo_met=10, slo_total=10),
        row(key="official:1", source="official", id=1, label="official run",
            j_per_req=200.0, slo_met=7, slo_total=10),
        row(label="<b>&x", j_per_req=150.0, slo_met=9, slo_total=10),
        row(key="local:gone", id="gone", label="gone", j_per_req=None),   # missing
        row(key="local:zero", id="zero", label="zero",
            j_per_req=100.0, slo_total=0, slo_met=0),                     # zero total
    ]
    svg = scatter_j_vs_slo(rows)
    texts = chart_texts(svg)
    assert_svg(svg)
    assert "short" in texts and "official run" in texts
    assert "J / request" in texts and "SLO met" in texts
    assert "upper-left = better" in texts
    # '<b>&x' must appear escaped (&lt;b&gt;&amp;x), never raw
    assert "&lt;b&gt;&amp;x" in svg
    assert "<b>" not in svg
    assert "&lt;b&gt;&amp;x" in texts or "<b>&x" in texts  # ET decodes .text


def test_scatter_long_label_truncated():
    svg = scatter_j_vs_slo([row(label="x" * 40, j_per_req=50.0,
                                slo_met=5, slo_total=10)])
    el = assert_svg(svg)
    labels = [t.text for t in el.iter("text") if (t.text or "").startswith("xx")]
    assert labels and max(len(t) for t in labels) <= 18


# ------------------------------------------------------------------ eta_bars

def test_eta_bars_values_and_baseline():
    rows = [
        row(label="cand", eta_vs_base=0.89,
            j_per_req=90.0, slo_met=9, slo_total=10),
        row(key="official:1", source="official", id=1, label="off",
            eta_vs_base=0.8901, S=0.6),
        row(key="local:skip", id="skip", eta_vs_base=None),  # skipped
    ]
    svg = eta_bars(rows)
    assert_svg(svg)
    texts = chart_texts(svg)
    assert "0.890" in texts          # known 3-decimal value appears
    assert "baseline" in texts
    assert "L cand" in texts and "O off" in texts  # source prefixes
    assert 'stroke-dasharray="4 3"' in svg          # dashed line at 1.0


def test_eta_bars_caps_at_25_rows():
    rows = [row(key=f"local:r{i}", id=f"r{i}", label=f"run{i}",
                started="2026-10-24T10:00:00", eta_vs_base=1.0 + i * 0.01)
            for i in range(30)]
    svg = eta_bars(rows)
    el = assert_svg(svg)
    # height grows with row count: 26 top margin + 25 * 22 + 14 = 590
    assert el.get("viewBox").endswith("590")
    rects = [e for e in el.iter("rect")]
    assert len(rects) == 25


def test_eta_bars_escaping_wellformed():
    svg = eta_bars([row(label="<b>&x", eta_vs_base=0.5)])
    assert_svg(svg)  # parses => escaping keeps the XML well-formed
    assert "&lt;b&gt;&amp;x" in svg
    assert "<b>" not in svg


# --------------------------------------------------------------- power_chart

def _trace(n, step=10.0):
    return [[i * step, 300.0 + i] for i in range(n)]


def test_power_chart_with_bounds():
    svg = power_chart(
        _trace(50),
        {"ramp": [0, 100], "peak": [100, 300], "quiet": [300, 400]},
    )
    assert_svg(svg)
    texts = chart_texts(svg)
    assert "ramp" in texts and "peak" in texts and "quiet" in texts
    assert "drain" in texts                          # drain after quiet end
    assert "seconds" in texts and "watts" in texts
    assert "<polyline" in svg
    assert 'fill-opacity="0.12"' in svg


def test_power_chart_bounds_missing_key():
    svg = power_chart(_trace(20), {"ramp": [0, 50]})  # peak/quiet missing
    assert_svg(svg)
    texts = chart_texts(svg)
    assert "ramp" in texts and "peak" not in texts and "quiet" not in texts


def test_power_chart_downsample():
    svg = power_chart(_trace(2000), None)
    el = assert_svg(svg)
    polyline = next(e for e in el.iter("polyline"))
    n_points = len(polyline.get("points").split())
    assert n_points <= 500


def test_power_chart_bad_input():
    for bad in (None, [], [[1]], "x", [[None, None]]):
        svg = power_chart(bad, {})
        assert_svg(svg)
        assert "no data" in chart_texts(svg)


def test_power_chart_custom_size():
    el = assert_svg(power_chart(_trace(10), {}, width=500, height=180))
    assert el.get("viewBox").startswith("0 0 500 180")
