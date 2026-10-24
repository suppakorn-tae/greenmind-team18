# Interfaces, phase 2 (official log, history, dashboard)   — extends INTERFACES.md

Rules for everyone: Python 3.12, venv `/home/researcher/grader/.venv/bin/python`, stdlib only (no pip), never edit files you don't own,
do not touch the GPU, do not contact localhost:8000. Run tests with `cd /home/researcher/grader && .venv/bin/python -m pytest <file> -q`.
Real data shapes: `fixtures/base/summary.json`, `fixtures/cand/summary.json` (+ records/items/power.csv), `gm/score.py`, `gm/store.py`.

## 1. Summary additions (already implemented in gm/score.summarize)
`n_arrivals`, `slo_turn{on_time,total}`, `slo_arrival{on_time,total}`, `slo_met`/`slo_total` (= per-arrival, official-comparable),
`p95_ttft_s` (raw seconds), `accuracy_pct` (micro %, 0-100), `j_per_req` (window joules ÷ arrivals), plus all older keys
(`eta`, `energy_J`, `s_T`, `s_L`, `p95_tpot_ms`, `class_acc`, `class_n`, `by_phase`, `by_class`, `bounds`, `phase_energy_J`, …).
Energy fields are None without NVML → treat None everywhere as "n/a", never crash.
score dict keys: `S, A, R, s_E, eta_ratio, class_factor, weakest_class, class_ratio{cls:ratio}, baseline_id` (S may be None = n/a).
A baseline run's score: {"S":0.0,"is_baseline":true,"eta_ratio":1.0,"R":1.0,"A":1.0,"s_E":0.0}.
Progress file while a run executes: `runs/<id>/progress.json` = {"items_done":int,"items_total":int,"started_at":iso,"updated_at":iso}.

## 2. gm/official.py   (SQLite table `official` inside the SAME file as the store: `store.db_path`)
```python
COLUMNS = ["id","track","started_at","model","label","notes","S","A","R","s_E","s_T","s_L","eta_vs_base","j_per_req",
           "p95_ttft_s","p95_tpot_ms","slo_met","slo_total","accuracy_pct","local_run_id","created_at"]
def init_table(store) -> None                       # CREATE TABLE IF NOT EXISTS; every function below calls it first
def add(store, track, started, model, S, A, R, s_E, s_T, s_L, eta_vs_base, j_per_req, p95_ttft_s, p95_tpot_ms,
        slo_met, slo_total, accuracy_pct, label="", notes="", local_run_id=None) -> int   # returns new id; validates track in val|final;
                                                    # started accepts "YYYY-MM-DD HH:MM:SS" or ISO → stored ISO 'YYYY-MM-DDTHH:MM:SS'
                                                    # local_run_id (id/prefix/label) is resolved with store.resolve and must be the same track
def get(store, official_id) -> dict                 # ValueError if missing
def list_official(store, track=None) -> list[dict]  # newest first by started_at then id
def link(store, official_id, run_id) -> str         # resolve run (store.resolve), same track else ValueError, sets local_run_id, returns full run id
def delete(store, official_id) -> None
def format_official_list(rows) -> str               # fixed-width table: ID TRACK STARTED LABEL S A R sE sT sL ETA J/REQ P95TTFT P95TPOT SLO ACC% LINK
```
Multiple official rows may link to one local run; an official row links to at most one run. Unknown/None metrics print '-'.

## 3. gm/history.py   (pure functions + one loader)
ROW = flat dict, identical keys for local and official rows (missing → None):
```
key            "local:<run_id>" | "official:<id>"
source         "local" | "official"
id             run_id (str) | official id (int)
track, label, started (ISO str), status ("done"|"running"|"interrupted"|"failed" ; official → "done")
model, power_limit (str|None, from server.nvidia_smi["power.limit"]), flags (str|None: vllm_cmdline minus the interpreter/path, e.g. "serve CMKL/... --max-model-len 32768 ..."),
server_summary (one-line text: "<model> | <power_limit> W | <key flags>"; key flags = --max-model-len, --dtype, --quantization, --max-num-seqs, --speculative-config, --gpu-memory-utilization, --enable-prefix-caching, --kv-cache-dtype when present),
is_baseline (bool; official: the row linked to the local baseline run of its track, else the earliest official row of the track whose label starts with "baseline"; False otherwise),
pair_key       key of the linked counterpart (local<->official) or None
S, A, R, s_E, s_T, s_L, eta_vs_base, j_per_req, p95_ttft_s, p95_tpot_ms, slo_met, slo_total, accuracy_pct   # official column set
slo_turn_met, slo_turn_total   # local only (per-turn view); official → None
eta            # absolute on-time req/J (local only)
group, repeat_k   # repeats: label "X#3" → group "X", repeat_k 3; else both None
notes
```
Local mapping: S,A,R,s_E,eta_vs_base come from run["score"] (S, A, R, s_E, eta_ratio); s_T,s_L,j_per_req,p95_ttft_s,p95_tpot_ms,slo_met,slo_total,accuracy_pct,eta from run["summary"].
Non-done local runs: metrics None.
```python
def local_row(run: dict) -> dict                        # run = Store.get_run() dict
def official_row(orow: dict, local_baseline_id: str|None, linked_run_key..) -> dict   # see docstring; implementer decides exact signature but build_rows wires it
def build_rows(runs: list[dict], officials: list[dict], track=None) -> list[ROW]   # newest first (by started), interleaved; sets is_baseline + pair_key
def load_rows(store, track=None) -> list[ROW]           # store.list_runs(track, limit=10**6) + official.list_official(store, track) → build_rows
def parse_repeat(label) -> (group|None, k|None)         # regex r"^(.*)#(\d+)$"
def group_repeats(rows) -> list[GROUP]                  # groups with >=2 local done members, same track+group
GROUP = {"track","group","n","members":[keys],"mean":{...},"min":{...},"max":{...},"spread_eta":float|None}
        # metrics in mean/min/max: S, eta_vs_base, eta, j_per_req, slo_met_frac (slo_met/slo_total), accuracy_pct, p95_ttft_s, p95_tpot_ms
        # spread_eta = (max-min)/mean of absolute eta (None if mean is 0/None)
def noise_spread(rows, default=0.10) -> dict            # {"spread": max spread_eta over groups else default, "observed": bool, "groups": n}
def within_noise(a, b, spread) -> bool                  # |a-b| / max(|b|, tiny) <= spread  (a,b = eta values or eta_vs_base values; None → False)
def mark_noise(rows, spread) -> None                    # sets row["within_noise"] = within_noise(row.eta_vs_base, baseline eta_vs_base (1.0 for local baseline), spread) for non-baseline rows of the same track; baseline row → None
def calibration(rows, track) -> dict
    {"pairs":[{"official_key","local_key","label","metrics":[{"name","label","local","official","ratio","flag"}]}],
     "rank_agreement":{"n_pairs":int,"agree":int,"total":int,"text":str}}
    # metrics, in order: j_per_req, slo_met_frac (slo_met/slo_total, 0-1), p95_ttft_s, p95_tpot_ms, accuracy_pct, eta_vs_base
    # ratio = local/official (None if either missing or official == 0); flag: "green" if |ratio-1|<=0.15, "amber" if <=0.30, "red" otherwise, "na" if ratio None
    # rank agreement: over all pairs-of-pairs, do local and official order the two configs the same way by eta_vs_base (tie = agree)? text e.g. "2 of 3 pairs agree" or "need >= 2 pairs"
def best_rows(rows, track) -> {"local": ROW|None, "official": ROW|None}   # highest S (None S ignored), ties → higher eta_vs_base
def format_history(rows, groups=None, spread=None) -> str   # fixed-width text table for the CLI, columns: SRC LABEL STARTED S A R sE sT sL ETA/BASE J/REQ P95TTFT P95TPOT SLO ACC% BASE PAIR CONFIG ; groups appended as "mean (min–max)" lines; rows with within_noise get "~" marker
```

## 4. Dashboard (phase 2)
`gm/dashboard_charts.py` (pure, returns SVG strings, no JS, theme via CSS vars `var(--fg)`, `var(--grid)`, `var(--local)`, `var(--official)`, `var(--accent)`):
```python
def s_over_time(rows) -> str      # x = started time, y = S; local = circles, official = diamonds; legend; handles <2 points / all None (return a small "no data" svg)
def scatter_j_vs_slo(rows) -> str # x = j_per_req (J/request), y = slo_met/slo_total (0-1); each row a dot with its label; "upper-left = better" note
def eta_bars(rows) -> str         # horizontal bars of eta_vs_base per row (label left, value right), a vertical line at 1.0
def power_chart(power, bounds, width=720, height=220) -> str   # power = [[t_s, watts],...]; bounds = {"ramp":[a,b],...}; shaded phase bands + labelled axes; None/[] → "no data"
```
`gm/dashboard.py`:
```python
def build_state(store, track) -> dict    # see JSON below
def build_run_detail(store, run_id) -> dict
def serve(store, host="0.0.0.0", port=8050)     # http.server ThreadingHTTPServer; reads sqlite live
def export_static(store, path) -> Path          # one self-contained html with data embedded (both tracks + every run detail)
```
HTTP: `/` page · `/api/state?track=val|final` · `/api/runs` (all local ROWs) · `/api/official` (all official ROWs) · `/api/run/<id>` · `/runs/<id>/report.html` (and other files in that run dir, whitelist: report.html, summary.json, power.csv) · 404 JSON otherwise. Never serve paths outside runs/. Sends `Cache-Control: no-store`.

state JSON (`/api/state?track=val`):
```json
{"track":"val","generated_at":"2026-10-24T12:00:00",
 "progress": null,
 "baselines":{"local":ROW|null,"official":ROW|null},
 "best":{"local":ROW|null,"official":ROW|null},
 "history":[ROW + {"within_noise":bool|null}],
 "groups":[GROUP], "noise":{"spread":0.1,"observed":false,"groups":0},
 "calibration":{"pairs":[...],"rank_agreement":{...}},
 "charts":{"s_over_time":"<svg…>","scatter":"<svg…>","eta_bars":"<svg…>"}}
```
progress when a local run has status "running" and its progress.json updated < 60 s ago: {"run_id","label","track","started_at","elapsed_s","items_done","items_total","stale":false}; if updated longer ago: same dict with "stale":true.
run detail JSON (`/api/run/<id>`): {"row":ROW,"summary":{...},"score":{...},"server":{...},"settings":{...},"baseline_id":str|null,
 "class_rows":[{"cls","acc","n","base_acc","ratio","checked":bool,"flag":"ok"|"low"}],   # flag "low" when checked and ratio < 0.75
 "fail_by_phase":{"ramp":{reason:count},...}, "tokens_by_class":{cls:{"mean","p95"}}, "power_svg":"<svg…>", "report_url":"/runs/<id>/report.html"}
