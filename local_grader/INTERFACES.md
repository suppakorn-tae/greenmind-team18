# Interfaces (source of truth for parallel work)

Python 3.12, venv `/home/researcher/grader/.venv/bin/python`. Stdlib only (+httpx, pynvml). Never edit files you don't own.
Read `gm/config.py`, `gm/score.py` (summarize/score), `gm/slo.py`, `tests/make_fixture.py` for the real data shapes.
Sample run folders already exist: `fixtures/base/` and `fixtures/cand/` (records.jsonl, items.jsonl, power.csv, summary.json).

## Run folder  runs/<run_id>/
- `records.jsonl`  one raw row per request (see tests/make_fixture.py for every key): id, cls, turn, sched_t, sent, launch_lag,
  phase, ttft_any, ttft_content, span_any, span_content, tpot_chunk_p95, prompt_tokens, completion_tokens, finish, has_usage,
  error, e2e, reasoning_chars, done, n_tool_calls, answer_tail.   (SLO verdicts are NOT stored; use `gm.slo.effective/evaluate`.)
- `items.jsonl`    {id, cls, correct, turns, final_turn}
- `power.csv`      header `t_s,watts,joules_cum` (t_s relative to trace origin t0)
- `summary.json`   output of `gm.score.summarize(...)` (keys: requests,on_time,on_time_frac,fail_reasons,s_T,s_L,s_L_ttft,s_L_tpot,
  p50/p95_ttft_over_slo,p50/p95_tpot_ms,completion_tokens_mean/p95,total_completion_tokens,cutoffs,by_phase{phase:group},
  by_class{cls:group},class_acc,class_n,macro_acc,bounds{phase:[a,b]},launch{max_launch_lag_ms,n_late_gt_100ms,max_loop_lag_ms},
  energy_raw,energy_ok,energy_J,window_s,avg_power_W,idle_W,eta,tokens_per_J,phase_energy_J{ramp,peak,quiet,drain}).
  group = {n,on_time,fails{reason:count},p50_ttft_ratio,p95_ttft_ratio,p50_tpot_ms,p95_tpot_ms,completion_tokens_mean,completion_tokens_p95,cutoffs}
  Energy fields are None when NVML was unavailable → show "n/a"/"—", never crash.
- `report.html`    single self-contained file (no external resources).

## score dict (gm.score.score)  — stored as score_json
{eta_ratio, s_E, R, A, class_factor, weakest_class:[cls,ratio]|None, class_ratio{cls:ratio}, S, baseline_id}
`{}` or {"S": None} when no baseline / no energy ⇒ display S = "n/a". The baseline run itself: {"S": 0.0, "is_baseline": true}.

## gm/store.py  (owner: Agent C)
```python
class Store:
    def __init__(self, root=None)      # root default = env GRADER_HOME or the grader/ dir; uses root/runs/ and root/runs.sqlite
    def create_run(self, track, label, notes, server: dict, settings: dict) -> str     # id YYYYmmdd-HHMMSS-<label slug>; unique (suffix -2, -3 on clash); makes folder; status 'running'
    def run_dir(self, run_id) -> Path
    def finish_run(self, run_id, status, summary: dict | None)   # status in done|interrupted|failed; sets ended_at
    def set_score(self, run_id, score: dict | None)
    def get_run(self, run_id) -> dict            # row with *_json columns decoded into keys server, settings, summary, score; is_baseline as bool
    def list_runs(self, track=None, limit=20) -> list[dict]   # newest first
    def resolve(self, ref) -> str                # exact id, else unique prefix or unique label match; ValueError if none/ambiguous
    def set_baseline(self, run_id)               # only status=done; unsets the track's previous baseline atomically; ValueError otherwise
    def get_baseline(self, track) -> dict | None
    def write_jsonl(self, run_id, name, rows); def read_jsonl(self, run_id, name) -> list[dict]
    def write_json(self, run_id, name, obj);   def read_json(self, run_id, name)
def format_list(rows) -> str     # table: id, track, label, status, base*, S, on-time, eta
def format_show(run) -> str      # multi-line key metrics + score
```
SQLite table `runs`: id TEXT PK, track, label, notes, status, started_at, ended_at, is_baseline INT, server_json, settings_json, summary_json, score_json.
Exactly one baseline per track.

## gm/report.py  (owner: Agent B)
```python
def build_report(store, run_id) -> Path          # writes runs/<id>/report.html; reads summary/records/items/power.csv + run row
def diff_text(store, id_a, id_b) -> str          # two-column key metrics + deltas + per-class deltas
```
