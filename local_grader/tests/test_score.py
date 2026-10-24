"""Unit tests for gm.score (accuracy factor / s_E / S / summarize). Owned by Agent D."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from gm import config, score
import make_fixture

ALL_ACC = {c: 1.0 for c in config.CLASS_MIX}          # baseline accuracy 1.0 for every class
LOSSES = [0, 2, 4, 6, 8, 10, 12, 15]
LOSS_A = [1.00, 1.00, 0.94, 0.77, 0.56, 0.33, 0.14, 0.05]   # expected score.accuracy_factor(...)[0]
ENERGY = [(1, 0.00), (1.25, 0.20), (1.5, 0.37), (2, 0.63), (2.5, 0.83), (3, 1.00)]  # (ratio, expected s_E)


def _read_jsonl(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_accuracy_factor_relative_loss():
    for loss, want in zip(LOSSES, LOSS_A):
        acc = {c: 1.0 - loss / 100.0 for c in config.CLASS_MIX}   # relative loss applied to all classes
        A, R, weakest, cf = score.accuracy_factor(acc, dict(ALL_ACC))
        assert A == pytest.approx(want, abs=0.005), f"relative loss {loss}% -> A {A}"


def test_s_energy():
    for ratio, want in ENERGY:
        assert score.s_energy(ratio) == pytest.approx(want, abs=0.005)
    assert score.s_energy(3.0) == 1.0            # ratio == K_ENERGY -> full
    assert score.s_energy(100.0) == 1.0          # ratio > K_ENERGY clips to 1
    assert score.s_energy(0.5) == 0.0            # ratio < 1 clips to 0


def test_weakest_class_and_class_factor():
    base = dict(ALL_ACC)
    acc = dict(base)
    acc["thai"] = 0.625                            # thai at 62.5% of baseline
    A, R, weakest, cf = score.accuracy_factor(acc, base)
    assert cf == pytest.approx(0.5, abs=1e-9)   # class factor from the smoothed ratio
    assert weakest == ("thai", 0.625)


def test_long_not_checked_has_unit_class_factor():
    base = dict(ALL_ACC)
    acc = dict(base)
    acc["long"] = 0.5                             # long is not in CHECKED -> factor must stay 1.0
    A, R, weakest, cf = score.accuracy_factor(acc, base)
    assert cf == 1.0


def test_zero_baseline_ignored_in_R():
    base = {c: 1.0 for c in config.CLASS_MIX}
    base["long"] = 0.0                           # baseline accuracy 0 -> ignored in the macro mean
    acc = {c: (0.5 if c in ("chat", "rag") else 1.0) for c in config.CLASS_MIX}
    A, R, weakest, cf = score.accuracy_factor(acc, base)
    # over the 5 classes with a real baseline: mean acc = (0.5+0.5+1+1+1)/5 = 0.8
    assert R == pytest.approx(0.8)


def test_p95_nearest_rank():
    assert score.p95(list(range(1, 101))) == 95
    assert score.p95([1]) == 1
    assert score.p95([]) is None
    assert score.p95([1.0, math.inf, 3.0]) == math.inf   # inf handled via sort


def test_score_s_formula_and_missing_eta():
    baseline = {"eta": 1.0, "class_acc": dict(ALL_ACC), "s_T": 0.5, "s_L": 0.5}
    summary = {"eta": 2.0, "class_acc": dict(ALL_ACC), "s_T": 0.8, "s_L": 0.9}
    got = score.score(summary, baseline)
    A, R, weakest, cf = score.accuracy_factor(summary["class_acc"], baseline["class_acc"])
    sE = score.s_energy(summary["eta"] / baseline["eta"])
    want = 100 * A * (sE ** 0.6) * (summary["s_T"] ** 0.2) * (summary["s_L"] ** 0.2)
    assert got["S"] == pytest.approx(want)
    assert got["eta_ratio"] == pytest.approx(2.0)
    # missing / eta-less baseline or summary -> empty score dict
    assert score.score(summary, {}) == {}
    assert score.score(summary, None) == {}
    assert score.score(summary, {"class_acc": dict(ALL_ACC)}) == {}
    assert score.score({"class_acc": dict(ALL_ACC)}, baseline) == {}


def test_summarize(tmp_path):
    s = make_fixture.make_fixture(tmp_path / "r")
    run = tmp_path / "r"
    records = _read_jsonl(run / "records.jsonl")
    items = _read_jsonl(run / "items.jsonl")
    # each_turn: one scored row per raw record
    assert s["requests"] == len(records)
    assert s["on_time"] <= s["requests"]
    assert sum(s["by_phase"][p]["n"] for p in config.PHASES) == s["requests"]
    assert 0.0 <= s["s_T"] <= 1.0
    assert 0.0 <= s["s_L"] <= 1.0

    # episode_once: one scored row per item
    s2 = score.summarize(
        records, items, s["energy_raw"], config.resolve_settings({"EPISODE_TURNS_SCORED": "episode_once"}),
        s["bounds"], s["launch"])
    assert s2["requests"] == len(items)
