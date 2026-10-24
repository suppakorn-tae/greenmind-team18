"""NVML energy meter + server/GPU context capture. `pynvml` is a module attribute so tests can monkeypatch it."""
import bisect, csv, subprocess, sys, threading, time
from pathlib import Path

try:
    import pynvml
except Exception:
    pynvml = None


class EnergyMeter:
    def __init__(self, gpu=0, interval=0.1):
        self.ok, self.samples, self.interval = False, [], interval
        self._stop = threading.Event()
        if pynvml is None:
            print("!! nvidia-ml-py not installed — energy will be null", file=sys.stderr)
            return
        try:
            pynvml.nvmlInit()
            self.h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.ok = True
        except Exception as e:
            print(f"!! NVML energy counter unavailable ({e}) — energy will be null; run on the GPU VM", file=sys.stderr)

    def _mj(self):
        return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.h)

    def idle_power(self, seconds=5.0):
        if not self.ok:
            return None
        e0, t0 = self._mj(), time.monotonic()
        time.sleep(seconds)
        return (self._mj() - e0) / 1000 / (time.monotonic() - t0)

    def start(self, t0):
        """Begin sampling; `t0` = monotonic time of the trace origin. Joules are relative to this call."""
        if not self.ok:
            return
        self.t0, self.e0 = t0, self._mj()
        self.samples = [(time.monotonic() - t0, 0.0, 0.0)]
        self.th = threading.Thread(target=self._run, daemon=True)
        self.th.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append((time.monotonic() - self.t0, pynvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0,
                                     (self._mj() - self.e0) / 1000.0))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def stop(self):
        """Stop sampling; → (t_rel, joules) read exactly at stop time (the window end), or None."""
        if not self.ok:
            return None
        self._stop.set()
        self.th.join()
        t, j = time.monotonic() - self.t0, (self._mj() - self.e0) / 1000.0
        self.samples.append((t, 0.0, j))
        return t, j

    def joules_at(self, t):
        """Cumulative joules at trace time t (linear interpolation between samples, clamped)."""
        if not self.samples:
            return None
        ts = [s[0] for s in self.samples]
        i = bisect.bisect_left(ts, t)
        if i <= 0:
            return self.samples[0][2]
        if i >= len(ts):
            return self.samples[-1][2]
        (ta, _, ja), (tb, _, jb) = self.samples[i - 1], self.samples[i]
        return ja + (jb - ja) * (t - ta) / (tb - ta) if tb > ta else jb

    def joules_between(self, a, b):
        ja, jb = self.joules_at(a), self.joules_at(b)
        return None if ja is None else max(0.0, jb - ja)

    def write_csv(self, path):
        with Path(path).open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "watts", "joules_cum"])
            for t, p, j in self.samples:
                w.writerow([round(t, 3), round(p, 1), round(j, 2)])

    def raw(self, bounds, end, first_send, idle_W):
        """Everything `score.summarize` needs from the meter (stored in the summary so `rescore` can switch
        ENERGY_WINDOW): both candidate windows + per-phase joules. `end` = (t_rel, joules) from stop()."""
        if not self.ok or end is None:
            return {"ok": False}
        t_end, j_end = end
        phase_J = {p: self.joules_between(a, min(b, t_end)) if a < t_end else 0.0 for p, (a, b) in bounds.items()}
        quiet_end = bounds["quiet"][1]
        phase_J["drain"] = self.joules_between(quiet_end, t_end) if t_end > quiet_end else 0.0
        js = self.joules_at(first_send)
        return {"ok": True, "idle_W": idle_W, "phase_J": phase_J,
                "windows": {"first_arrival_to_last_done": {"J": j_end, "s": t_end},
                            "first_send_to_last_done": {"J": j_end - js, "s": t_end - first_send}}}


def _is_server(pid):
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").lower()
    except OSError:
        return False
    return "vllm" in cmd


def other_compute_pids(gpu=0):
    """PIDs with a compute context on the GPU that don't look like the vLLM server. None if NVML is unavailable."""
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
    except Exception:
        return None
    return [p.pid for p in procs if not _is_server(p.pid)]


def server_context():
    """nvidia-smi power limit / clocks / driver + the full `vllm serve` command line. Never raises."""
    ctx = {}
    q = "name,driver_version,power.limit,power.draw,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,persistence_mode"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        if out:
            ctx["nvidia_smi"] = dict(zip(q.split(","), [v.strip() for v in out[0].split(",")]))
    except Exception as e:
        ctx["nvidia_smi_error"] = repr(e)[:200]
    try:
        ps = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True, timeout=10).stdout.splitlines()
        ctx["vllm_cmdline"] = next((l.strip() for l in ps if "vllm" in l and "serve" in l and "ps -eo" not in l), None)
    except Exception as e:
        ctx["ps_error"] = repr(e)[:200]
    return ctx
