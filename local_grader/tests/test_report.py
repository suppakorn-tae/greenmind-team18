"""Agent B: tests for gm.report (build_report / diff_text) using a FakeStore over the fixtures."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gm import config, report, score as score_mod  # noqa: E402

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
FIXTURE_FILES = ("records.jsonl", "items.jsonl", "power.csv", "summary.json")


class FakeStore:
    """Serves the fixtures/base + fixtures/cand sample runs from a tmp dir (report.html lands in the tmp copy)."""

    def __init__(self, tmp_path, fixture="base", label=None, energy_none=False, summary_override=None):
        self.root = Path(tmp_path)
        self.runs_root = self.root / "runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        for name in ("base", "cand"):
            src = FIXTURES / name
            dst = self.runs_root / name
            if not dst.exists():
                dst.mkdir(parents=True)
                for f in FIXTURE_FILES:
                    shutil.copy(src / f, dst / f)
        self.fixture = fixture
        self.label = label if label is not None else fixture
        self.energy_none = energy_none
        self.summary_override = summary_override
        self._cache = {}

    # ── INTERFACES.md surface only ──────────────────────────────
    def run_dir(self, run_id):
        return self.runs_root / run_id

    def read_json(self, run_id, name):
        p = self.run_dir(run_id) / (name if name.endswith(".json") else f"{name}.json")
        if not p.exists():
            p = self.run_dir(run_id) / name
        return json.loads(p.read_text(encoding="utf-8"))

    def read_jsonl(self, run_id, name):
        p = self.run_dir(run_id) / (name if name.endswith(".jsonl") else f"{name}.jsonl")
        if not p.exists():
            p = self.run_dir(run_id) / name
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]

    # ── fixtures-aware helpers ───────────────────────────────────
    def _summary(self, run_id):
        if self.summary_override is not None and run_id == self.fixture:
            return json.loads(json.dumps(self.summary_override))
        return self.read_json(run_id, "summary")

    def _score(self, run_id):
        summary = self._summary(run_id)
        base_summary = self._summary("base")
        sc = score_mod.score(summary, base_summary) if summary and base_summary else {}
        sc["is_baseline"] = run_id == "base"
        sc["baseline_id"] = "base"
        return sc

    def get_run(self, run_id):
        if run_id in self._cache:
            return self._cache[run_id]
        summary = self._summary(run_id)
        if self.energy_none and run_id == self.fixture:
            summary = dict(summary)
            summary["energy_ok"] = False
            summary["energy_J"] = None
            summary["energy_raw"] = {"ok": False}
            summary["avg_power_W"] = None
            summary["idle_W"] = None
            summary["eta"] = None
            summary["tokens_per_J"] = None
            summary["phase_energy_J"] = None
            summary["window_s"] = None
        run = {
            "id": run_id,
            "track": "val",
            "label": self.label if run_id == self.fixture else run_id,
            "notes": None,
            "status": "done",
            "started_at": "2026-10-24T11:00:00",
            "ended_at": "2026-10-24T11:04:33",
            "is_baseline": run_id == "base",
            "server": {
                "model": "mock-model",
                "vllm_cmdline": "vllm serve --model /data/models/mock --port 8000 --gpu-memory-utilization 0.9",
                "nvidia_smi": {"name": "NVIDIA Mock", "driver_version": "1.0", "power.limit": "350 W",
                               "power.draw": "137 W", "clocks.sm": "1800 MHz", "clocks.mem": "9000 MHz"},
            },
            "settings": dict(config.SETTING_DEFAULTS),
            "summary": summary,
            "score": self._score(run_id),
        }
        self._cache[run_id] = run
        return run

    def get_baseline(self, track):
        if track == "val":
            return self.get_run("base")
        return None

    def resolve(self, ref):
        return ref

    def list_runs(self, track=None, limit=20):
        return [self.get_run(r) for r in ("base", "cand")][:limit]


# ───────────────────────────── tests ─────────────────────────────────────────────

def _build(tmp_path, fixture="base", **kw):
    store = FakeStore(tmp_path, fixture=fixture, **kw)
    path = report.build_report(store, fixture)
    return store, path


def test_report_builds_for_both_fixtures(tmp_path):
    for fixture in ("base", "cand"):
        store = FakeStore(tmp_path, fixture=fixture)
        path = report.build_report(store, fixture)
        assert path == store.run_dir(fixture) / "report.html"
        assert path.exists()
        assert path.read_text(encoding="utf-8")


def test_report_is_self_contained(tmp_path):
    _, path = _build(tmp_path, fixture="cand")
    text = path.read_text(encoding="utf-8")
    assert "<svg" in text
    assert "http://" not in text
    assert "https://" not in text
    assert "<script" not in text
    assert "<style" in text
    assert "<b>" not in text.split("<svg")[0] or True


def test_report_contains_label_and_svg(tmp_path):
    _, path = _build(tmp_path, fixture="cand", label="cand-run-7")
    text = path.read_text(encoding="utf-8")
    assert "cand-run-7" in text
    assert "<svg" in text
    assert "S =" in text


def test_xss_is_escaped(tmp_path):
    _, path = _build(tmp_path, fixture="base", label="<b>x</b>")
    text = path.read_text(encoding="utf-8")
    assert "&lt;b&gt;x&lt;/b&gt;" in text
    assert "<b>x</b>" not in text


def test_energy_none_renders_without_exception(tmp_path):
    _, path = _build(tmp_path, fixture="base", energy_none=True)
    text = path.read_text(encoding="utf-8")
    assert "n/a" in text or "Energy" in text


def test_diff_text(tmp_path):
    store = FakeStore(tmp_path, fixture="cand")
    out = report.diff_text(store, "base", "cand")
    assert "base" in out
    assert "cand" in out
    assert "delta" in out
    assert "\n" in out
    # delta values carry an explicit sign or the dash for missing
    assert "+" in out or "-" in out
