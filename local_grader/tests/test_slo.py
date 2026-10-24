"""Unit tests for gm.slo (TTFT/TPOT evaluation, effective rows). Owned by Agent D."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from gm import config, slo

SETTINGS = config.resolve_settings()


def good_record(**over):
    rec = {"id": 0, "cls": "chat", "phase": "ramp", "turn": 0, "ttft_any": 0.5, "ttft_content": 0.5, "span_any": 10.0, "span_content": 5.0,
           "tpot_chunk_p95": 0.01, "prompt_tokens": 1000, "completion_tokens": 500, "finish": "stop",
           "has_usage": True, "error": None}
    rec.update(over)
    return rec


def test_ttft_allowance():
    assert slo.ttft_allowance(300) == pytest.approx(0.575)
    assert slo.ttft_allowance(6000) == pytest.approx(2.0)
    assert slo.ttft_allowance(20000) == pytest.approx(5.5)


def test_evaluate_good_record():
    assert slo.evaluate(good_record(), SETTINGS) == []


def test_evaluate_failure_reasons_in_isolation():
    assert slo.evaluate(good_record(ttft_any=None), SETTINGS) == ["no_stream"]
    assert slo.evaluate(good_record(error="boom"), SETTINGS) == ["error"]
    assert slo.evaluate(good_record(has_usage=False), SETTINGS) == ["no_usage"]
    assert slo.evaluate(good_record(finish="length"), SETTINGS) == ["cutoff"]
    assert slo.evaluate(good_record(completion_tokens=8192), SETTINGS) == ["cutoff"]
    # ttft just above / just below the allowance (prompt_tokens=1000 -> allowance 0.75)
    assert slo.evaluate(good_record(ttft_any=0.75 * 1.001), SETTINGS) == ["ttft"]
    assert slo.evaluate(good_record(ttft_any=0.75 * 0.999), SETTINGS) == []
    # tpot just above / just below the SLO (span_any/(ct-1) with ct=101 -> tpot = span/100)
    assert slo.evaluate(good_record(completion_tokens=101, span_any=0.051 * 100), SETTINGS) == ["tpot"]
    assert slo.evaluate(good_record(completion_tokens=101, span_any=0.05 * 100), SETTINGS) == []
    # tpot skipped when completion_tokens <= 1
    assert slo.evaluate(good_record(completion_tokens=1, span_any=0.2), SETTINGS) == []


def test_tpot_of_and_ttFT_from():
    rec = good_record(completion_tokens=500, span_any=10.0, span_content=5.0, tpot_chunk_p95=0.01)
    assert slo.tpot_of(rec, SETTINGS) == pytest.approx(10.0 / 499.0)
    assert slo.tpot_of(rec, config.resolve_settings({"TPOT_MODE": "p95_over_tokens"})) == pytest.approx(0.01)
    # TTFT_FROM changes ttft only; tpot keeps the full token span (completion_tokens includes reasoning tokens)
    assert slo.tpot_of(rec, config.resolve_settings({"TTFT_FROM": "content_only"})) == pytest.approx(10.0 / 499.0)
    assert slo.ttft_of(rec, SETTINGS) == pytest.approx(0.5)
    assert slo.ttft_of(rec, config.resolve_settings({"TTFT_FROM": "content_only"})) == pytest.approx(0.5)


def test_effective_each_turn():
    records = [good_record(id=1, phase="ramp", turn=0), good_record(id=1, phase="peak", turn=1, ttft_any=5.0)]
    rows = slo.effective(records, SETTINGS)
    assert len(rows) == 2
    assert [r["turn"] for r in rows] == [0, 1]


def test_effective_episode_once():
    records = [good_record(id=7, phase="ramp", turn=0), good_record(id=7, phase="peak", turn=1, ttft_any=5.0)]
    rows = slo.effective(records, config.resolve_settings({"EPISODE_TURNS_SCORED": "episode_once"}))
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == 7
    assert r["phase"] == "ramp"            # phase of the first turn
    assert r["fails"] == ["ttft"]          # merged: on time only if all turns are
    assert r["slo_ok"] is False
