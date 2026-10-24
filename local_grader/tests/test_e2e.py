"""End-to-end: mock OpenAI server + fake time-based NVML. Full val-track replay, twice (300 W baseline, 120 W run).
Takes ~2 min (real-time open-loop replay of a 45 s trace, twice)."""
import argparse, asyncio, math, os, sys, time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import grader
from gm import energy
from gm.store import Store


class FakeNVML:
    """energy(mJ) = ∫ watts dt, with a settable constant power."""
    def __init__(self, watts):
        self.watts, self.t_last, self.mj = watts, time.monotonic(), 0.0
        self.procs = []

    def set_power(self, w):
        self._tick(); self.watts = w

    def _tick(self):
        now = time.monotonic(); self.mj += self.watts * 1000 * (now - self.t_last); self.t_last = now

    def nvmlInit(self): pass
    def nvmlDeviceGetHandleByIndex(self, i): return object()
    def nvmlDeviceGetTotalEnergyConsumption(self, h): self._tick(); return int(self.mj)
    def nvmlDeviceGetPowerUsage(self, h): return int(self.watts * 1000)
    def nvmlDeviceGetComputeRunningProcesses(self, h): return self.procs


def make_args(url, track="val", label="t", **kw):
    d = dict(track=track, label=label, notes="", url=url, model=None, force=False, gpu=0, warmup=1, settle=1.0)
    d.update(kw)
    return argparse.Namespace(**d)


@pytest.fixture
def fake_nvml(monkeypatch):
    fake = FakeNVML(300.0)
    monkeypatch.setattr(energy, "pynvml", fake)
    monkeypatch.setattr(energy, "server_context", lambda: {"nvidia_smi": {"power.limit": "300"}, "vllm_cmdline": "vllm serve mock"})
    return fake


def test_full_val_baseline_then_lower_power(mock_server, fake_nvml, tmp_path):
    st = Store(tmp_path)
    fake_nvml.set_power(300.0)
    base_id = asyncio.run(grader.run_track(make_args(mock_server.url, label="base"), st))
    st.set_baseline(base_id)
    grader.rescore(st, None, {})
    base = st.get_run(base_id)
    assert base["status"] == "done" and base["score"]["S"] == 0.0 and base["is_baseline"]

    fake_nvml.set_power(120.0)
    run_id = asyncio.run(grader.run_track(make_args(mock_server.url, label="low"), st))
    run = st.get_run(run_id)
    s, sc = run["summary"], run["score"]
    assert s["on_time"] == s["requests"] and s["requests"] >= 100        # every request on time (agent turns add more)
    assert s["class_acc"]["agent"] == 1.0 and s["class_acc"]["rag"] == 1.0
    assert s["s_T"] == 1.0 and s["s_L"] > 0.9
    assert s["launch"]["n_late_gt_100ms"] == 0, s["launch"]
    assert sc["eta_ratio"] == pytest.approx(2.5, rel=0.03)
    assert sc["s_E"] == pytest.approx(math.log(2.5) / math.log(3), abs=0.01)   # ≈ 0.83
    assert sc["A"] == 1.0 and 70 < sc["S"] <= 100
    assert sc["baseline_id"] == base_id
    d = Path(st.run_dir(run_id))
    for f in ("records.jsonl", "items.jsonl", "power.csv", "summary.json", "report.html"):
        assert (d / f).exists(), f
    assert st.get_run(run_id)["server"]["model"] == "mock-model"
    # rescore with a different calibration switch changes nothing material for the mock but must work from stored records
    grader.rescore(st, run_id, {"TTFT_FROM": "content_only", "EPISODE_TURNS_SCORED": "episode_once"})
    r2 = st.get_run(run_id)
    assert r2["settings"]["TTFT_FROM"] == "content_only" and r2["summary"]["requests"] == 100


def test_refuses_when_health_fails(fake_nvml, tmp_path):
    with pytest.raises(SystemExit):
        asyncio.run(grader.run_track(make_args("http://127.0.0.1:9"), Store(tmp_path)))


def test_refuses_foreign_gpu_process(mock_server, fake_nvml, tmp_path):
    fake_nvml.procs = [SimpleNamespace(pid=os.getpid())]     # this pytest process: a compute process that is not vllm
    st = Store(tmp_path)
    with pytest.raises(SystemExit) as e:
        asyncio.run(grader.run_track(make_args(mock_server.url), st))
    assert "--force" in str(e.value) and st.list_runs() == []


def test_no_nvml_runs_with_null_energy(mock_server, monkeypatch, tmp_path):
    monkeypatch.setattr(energy, "pynvml", None)
    monkeypatch.setattr(energy, "server_context", lambda: {})
    st = Store(tmp_path)
    rid = asyncio.run(grader.run_track(make_args(mock_server.url, settle=0), st))
    run = st.get_run(rid)
    assert run["status"] == "done" and run["summary"]["eta"] is None and run["score"]["S"] is None
