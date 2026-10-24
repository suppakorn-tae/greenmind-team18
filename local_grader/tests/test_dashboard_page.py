"""Agent F: tests for the dashboard page template (`gm.dashboard_page`)."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gm import dashboard_page  # noqa: E402

PAGE = dashboard_page.PAGE_HTML


def _script_body(html):
    """Return the body of the inline application script (not the export-data block)."""
    match = re.search(r"<script>(.*?)</script>", html, re.S)
    assert match is not None, "inline dashboard script is missing"
    return match.group(1)


def _export_block(html):
    match = re.search(
        r'<script id="export-data" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match is not None, "export-data block is missing"
    return match.group(1)


def _sample_export():
    hostile = "</script><b>boom</b>"
    hostile_comment = "<!-- <script>alert(1)</script> -->"
    state = {
        "track": "val",
        "generated_at": "2026-10-24T12:00:00",
        "progress": None,
        "baselines": {"local": None, "official": None},
        "best": {"local": None, "official": None},
        "history": [
            {
                "key": "local:run-1",
                "source": "local",
                "id": "run-1",
                "track": "val",
                "label": hostile,
                "started": "2026-10-24T11:00:00",
                "status": "done",
                "S": 1.25,
                "A": 0.9,
                "R": 1.0,
                "s_E": 0.7,
                "s_T": 0.8,
                "s_L": 0.9,
                "eta_vs_base": 1.05,
                "j_per_req": 349.6,
                "p95_ttft_s": 0.54,
                "p95_tpot_ms": 43.1,
                "slo_met": 87,
                "slo_total": 100,
                "accuracy_pct": 89.0,
                "is_baseline": False,
                "pair_key": "official:1",
                "server_summary": "mock | 350 W | --dtype bfloat16",
                "within_noise": True,
            }
        ],
        "groups": [],
        "noise": {"spread": 0.1, "observed": False, "groups": 0},
        "calibration": {
            "pairs": [],
            "rank_agreement": {"n_pairs": 0, "agree": 0, "total": 0, "text": "need >= 2 pairs"},
        },
        "charts": {
            "s_over_time": "<svg></svg>",
            "scatter": "<svg></svg>",
            "eta_bars": "<svg></svg>",
        },
    }
    detail = {
        "row": state["history"][0],
        "summary": {"energy_J": None},
        "score": {"S": 1.25},
        "server": {"model": "mock"},
        "settings": {},
        "baseline_id": None,
        "class_rows": [
            {"cls": "chat", "acc": 0.5, "n": 10, "base_acc": 0.9, "ratio": 0.55, "checked": True, "flag": "low"}
        ],
        "fail_by_phase": {"ramp": {"ttft": 2}, "peak": {"tpot": 1}},
        "tokens_by_class": {"chat": {"mean": 120.5, "p95": 240}},
        "power_svg": "<svg></svg>",
        "report_url": "/runs/run-1/report.html",
        "notes": hostile_comment,
    }
    return {
        "states": {"val": state, "final": dict(state, track="final", history=[], charts={})},
        "runs": {"run-1": detail},
    }


def test_page_is_complete_and_offline():
    assert isinstance(PAGE, str)
    assert PAGE.lower().startswith("<!doctype html>")
    lowered = PAGE.lower()
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "<link" not in lowered
    assert not re.search(r"<script[^>]*\ssrc\s*=", PAGE, re.I)
    assert PAGE.count("</script>") == 1


def test_required_ids_and_markers():
    for element_id in ("status", "best", "history", "calibration", "charts", "detail"):
        assert 'id="%s"' % element_id in PAGE
    assert "setInterval" in PAGE
    assert "XMLHttpRequest" in PAGE
    assert "EXPORT" in PAGE
    assert "function api(path, callback)" in PAGE
    assert "JSON.parse(node.textContent)" in PAGE


def test_render_live_is_page_html():
    assert dashboard_page.render_page(None) == PAGE
    assert dashboard_page.render_page() == PAGE


def test_export_embeds_round_tripping_json():
    page = dashboard_page.render_page(export=_sample_export())
    raw = _export_block(page)
    assert "</" not in raw
    assert "<!--" not in raw
    assert "</script>" not in raw
    restored = json.loads(raw.replace("<\\/", "</"))
    expected = _sample_export()
    assert restored == expected
    assert restored["states"]["val"]["history"][0]["label"] == "</script><b>boom</b>"
    assert restored["runs"]["run-1"]["notes"] == "<!-- <script>alert(1)</script> -->"


def test_hostile_data_does_not_end_the_script_early():
    page = dashboard_page.render_page(export=_sample_export())
    assert page.count("</script>") == PAGE.count("</script>") + 1
    raw = _export_block(page)
    assert raw.count("</") == 0
    assert raw.count("<!--") == 0


def test_export_mode_is_gated_and_has_no_extra_network_code():
    page = dashboard_page.render_page(export={"states": {}, "runs": {}})
    assert "var EXPORT = true;" in page
    assert "var EXPORT = false;" in PAGE
    assert re.search(r"if \(!EXPORT\) \{\s*setInterval", _script_body(page))
    assert "XMLHttpRequest" in page
    assert "EXPORT" in page


def test_js_is_conservative_es5_and_brackets_balance():
    js = _script_body(PAGE)
    for opening, closing in (("{", "}"), ("(", ")"), ("[", "]")):
        assert js.count(opening) == js.count(closing), "%s/%s unbalanced" % (opening, closing)
    assert "=>" not in js
    assert "`" not in js
    assert "async " not in js
    assert "await " not in js
    assert "fetch(" not in js
    assert "const " not in js
    assert "let " not in js


def test_expected_page_content():
    assert "GreenMind local grader" in PAGE
    assert "Best local S" in PAGE
    assert "Best official S" in PAGE
    assert "No run in progress" in PAGE
    assert "Run in progress:" in PAGE
    assert "Local baseline" in PAGE
    assert "Official baseline" in PAGE
    assert "Noise spread" in PAGE
    assert "No linked local/official pairs yet" in PAGE
    assert "S over time" in PAGE
    assert "J/request vs SLO met" in PAGE
    assert "η vs baseline" in PAGE
    assert "below 0.75 of baseline" in PAGE
