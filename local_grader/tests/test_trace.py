"""Unit tests for gm.trace (loading, counts, bounds, schema, freeze, stats, poisson). Owned by Agent D."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from gm import config, trace

LINE_KEYS = {"id", "t", "phase", "cls", "messages", "tools", "episode", "check"}
CHECK_TYPES = {"contains", "number", "nonempty", "toolcall"}
VAL_BOUNDS = {"ramp": (0.0, 10.0), "peak": (10.0, 25.0), "quiet": (25.0, 45.0)}
FINAL_BOUNDS = {"ramp": (0.0, 30.0), "peak": (30.0, 75.0), "quiet": (75.0, 135.0)}


def _by(items, key):
    out = {}
    for it in items:
        out[it[key]] = out.get(it[key], 0) + 1
    return out


def test_load_both_tracks():
    val, val_bounds, val_meta = trace.load("val")
    final, final_bounds, final_meta = trace.load("final")
    assert len(val) == 100 and len(final) == 300
    assert val_bounds == VAL_BOUNDS
    assert final_bounds == FINAL_BOUNDS
    assert val_meta["n"] == 100 and final_meta["n"] == 300


def test_class_and_phase_counts():
    val = trace.load("val")[0]
    final = trace.load("final")[0]
    assert _by(val, "cls") == {"chat": 30, "rag": 20, "agent": 20, "reason": 15, "thai": 10, "long": 5}
    assert _by(final, "cls") == {"chat": 90, "rag": 60, "agent": 60, "reason": 45, "thai": 30, "long": 15}
    assert _by(val, "phase") == {"ramp": 20, "peak": 60, "quiet": 20}
    assert _by(final, "phase") == {"ramp": 60, "peak": 180, "quiet": 60}


def test_phase_bounds_and_monotonic_t():
    val, val_bounds, _ = trace.load("val")
    final, final_bounds, _ = trace.load("final")
    for items, bounds in ((val, val_bounds), (final, final_bounds)):
        prev = -1.0
        for it in items:
            a, b = bounds[it["phase"]]
            assert a <= it["t"] < b
            assert it["t"] >= prev
            prev = it["t"]


def test_line_schema():
    for items in (trace.load("val")[0], trace.load("final")[0]):
        for it in items:
            assert set(it) == {"id", "t", "phase", "cls", "messages", "tools", "episode", "check"}
            assert isinstance(it["episode"], bool)
            assert it["episode"] == (it["cls"] == "agent")
            assert (it["tools"] is not None) == (it["cls"] == "agent")
            assert it["check"]["type"] in CHECK_TYPES


def test_deterministic_and_freeze(tmp_path):
    a, b1 = trace.build_trace(100, 0)
    b, b2 = trace.build_trace(100, 0)
    assert a == b and b1 == b2
    assert trace.build_trace(100, 1)[0] != a                    # different seed -> different trace

    paths = trace.freeze(0, force=False, root=tmp_path)        # first freeze into the empty root: ok
    with pytest.raises(FileExistsError):
        trace.freeze(0, force=False, root=tmp_path)            # second freeze refuses
    trace.freeze(0, force=True, root=tmp_path)                 # ... and succeeds with force=True

    lines = (tmp_path / "traces" / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    frozen = [json.loads(l) for l in lines if l.strip()]
    assert frozen == a
    assert all(json.dumps(json.loads(l), ensure_ascii=False) == l for l in lines)   # json round-trip


def test_long_context_prompts():
    final = trace.load("final")[0]
    assert trace.stats(final)["max_prompt_tokens_approx"] <= 32000


def test_poisson_arrivals():
    a, b1 = trace.build_trace(100, 0, "poisson")
    b, b2 = trace.build_trace(100, 0, "poisson")
    assert a == b and b1 == b2
    assert _by(a, "phase") == {"ramp": 20, "peak": 60, "quiet": 20}


def test_build_trace_independent_of_hash_seed():
    """Regression: greenbench built RAG vault ids in a set, so prompts depended on PYTHONHASHSEED."""
    import hashlib, json, os, subprocess, sys
    code = ("import hashlib, json, sys; sys.path.insert(0, %r); from gm import trace; "
            "it, _ = trace.build_trace(100, 0); print(hashlib.md5(json.dumps(it, sort_keys=True).encode()).hexdigest())"
            % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": str(h)}).stdout.strip() for h in (1, 2, 3)}
    assert len(outs) == 1 and "" not in outs
